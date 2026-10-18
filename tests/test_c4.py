"""C4 source provenance, isolation, real kg_query, and runner integration.

The 224-row tests use the existing in-memory index fixture, with original
fragment records built independently from per-document ground-truth facts.
They do not represent paid model runs or real document extraction.
"""
from __future__ import annotations

import asyncio
import copy
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from fake_lightrag import FakeRAG, import_operate
from acl_rag import ACLIndex, Mode, Scope, acting_as, install, set_c4_description_filter
from acl_rag.c4 import C4DescriptionFilter
from acl_rag.fragments import FragmentStore, MissingProvenanceError
from acl_rag.run import resume_rows, run_one

CORPUS = ROOT / "corpus/dev_cf0.3_s42"
OPERATE = import_operate()


def dev_fragments(store):
    """Create source records without reading any merged graph descriptions."""
    ft = json.loads((CORPUS / "facts.json").read_text())
    entities = {e["entity_id"]: e for e in ft["entities"]}
    facts = {f["fact_id"]: f for f in ft["facts"]}
    chunks, nodes, edges = {}, defaultdict(list), defaultdict(list)
    for line in (CORPUS / "manifest.jsonl").read_text().splitlines():
        doc = json.loads(line)
        cid = "chunk-" + doc["doc_id"]
        chunks[cid] = {"full_doc_id": doc["doc_id"],
                       "content": (CORPUS / "docs" / (doc["doc_id"] + ".txt")).read_text()}
        for eid in doc["entities"]:
            e = entities[eid]
            desc = " ".join(facts[f]["sentence"] for f in doc["fact_ids"]
                            if eid in facts[f]["mentions"])
            nodes[e["canonical_name"]].append({"source_id": cid,
                "description": desc or e["canonical_name"] + " is mentioned in this document.",
                "entity_type": e["type"]})
        for fid in doc["fact_ids"]:
            f = facts[fid]
            if "entity" in f["object"]:
                pair = (entities[f["subject"]]["canonical_name"],
                        entities[f["object"]["entity"]]["canonical_name"])
                edges[pair].append({"source_id": cid, "description": f["sentence"],
                                    "keywords": f["predicate"]})
    store.capture(chunks, [(nodes, edges)])


@pytest.fixture
def simple(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    public, executive = "u_public_1", "u_exec_1"
    docs = [{"doc_id": "public", "acl_users": [public, executive]},
            {"doc_id": "private", "acl_users": [executive]}]
    (corpus / "manifest.jsonl").write_text("\n".join(json.dumps(d) for d in docs))
    (corpus / "oracle_splits.json").write_text(json.dumps({
        "access_sets": {"AS0": {}, "AS3": {}},
        "user_to_access_set": {public: "AS0", executive: "AS3"}}))
    store = FragmentStore(tmp_path / "fragments.sqlite3", corpus, create=True)
    chunks = {"pub": {"full_doc_id": "public", "content": "public source"},
              "priv": {"full_doc_id": "private", "content": "restricted source"}}
    nodes = {"Project Atlas": [
        {"source_id": "pub", "description": "Project Atlas is active.", "entity_type": "PROJECT"},
        {"source_id": "priv", "description": "Project Atlas budget code is ZZ-1234-X.",
         "entity_type": "PROJECT"}]}
    edges = {("Project Atlas", "Team North"): [
        {"source_id": "pub", "description": "Team North runs Project Atlas.", "keywords": "runs"},
        {"source_id": "priv", "description": "Team North funds secret ZZ-1234-X.", "keywords": "ZZ-1234-X"}]}
    store.capture(chunks, [(nodes, edges)])
    acl = ACLIndex(corpus)

    async def get_chunks(ids):
        return [dict(chunks[c]) if c in chunks else None for c in ids]

    async def node(_):
        return {"source_id": "pub<SEP>priv"}

    async def edge(*_):
        return {"source_id": "pub<SEP>priv"}

    acl.bind(get_chunks, node, edge)
    yield corpus, store, acl, chunks, nodes, edges
    store.close()


def test_source_records_preserve_permissions_and_endpoint_provenance(simple):
    _, store, _, _, _, _ = simple
    records = store.lookup("node", "Project Atlas")
    assert len(records) == 2
    assert {r["doc_id"] for r in records} == {"public", "private"}
    assert "u_public_1" not in next(r for r in records if r["chunk_id"] == "priv")["acl_users"]
    assert len(store.lookup("node", "Team North")) == 2
    assert store.lookup("edge", ("Team North", "Project Atlas"))


def test_recapture_replaces_old_fragments_and_changes_cache_identity(simple):
    _, store, _, chunks, _, _ = simple
    before = store.fingerprint()
    store.capture({"priv": chunks["priv"]}, [({"Project Atlas": [
        {"source_id": "priv", "description": "The revised restricted fact."}]}, {})])
    assert store.fingerprint() != before
    assert "ZZ-1234-X" not in " ".join(r["description"] for r in store.lookup("node", "Project Atlas"))
    assert len(store.lookup("node", "Project Atlas")) == 2


def test_ambiguous_fragments_are_rejected_atomically(simple):
    _, store, _, chunks, _, _ = simple
    before = store.fingerprint()
    with pytest.raises(MissingProvenanceError, match="one input chunk"):
        store.capture(chunks, [({"Project Atlas": [
            {"source_id": "pub<SEP>priv", "description": "Already blended secret"}]}, {})])
    assert store.fingerprint() == before


def test_identical_text_keeps_independent_source_permissions(simple):
    _, store, acl, chunks, _, _ = simple
    text = "Project Atlas is active."
    store.capture(chunks, [({"Project Atlas": [
        {"source_id": cid, "description": text} for cid in chunks]}, {})])
    assert len(store.lookup("node", "Project Atlas")) == 2
    filt = C4DescriptionFilter(store, strategy="join")

    async def go():
        safe = await filt("node", "Project Atlas", {"source_id": "pub<SEP>priv"},
                          Scope(Mode.C4, "u_public_1", "AS0"), acl)
        full = await filt("node", "Project Atlas", {"source_id": "pub<SEP>priv"},
                          Scope(Mode.C4, "u_exec_1", "AS3"), acl)
        assert safe["description"] == full["description"] == text
        assert safe["source_id"] == "pub"
        assert full["source_id"] == "pub<SEP>priv"

    asyncio.run(go())


def test_public_filter_removes_restricted_text_and_metadata_without_mutating_graph(simple):
    _, store, acl, _, _, _ = simple
    filt = C4DescriptionFilter(store, strategy="join")
    original = {"source_id": "pub<SEP>priv", "description": "irreversible blend ZZ-1234-X",
                "keywords": "runs, ZZ-1234-X", "file_path": "public<SEP>private"}

    async def go():
        public = await filt("node", "Project Atlas", original, Scope(Mode.C4, "u_public_1", "AS0"), acl)
        executive = await filt("node", "Project Atlas", original, Scope(Mode.C4, "u_exec_1", "AS3"), acl)
        edge = await filt("edge", ("Team North", "Project Atlas"), original,
                          Scope(Mode.C4, "u_public_1", "AS0"), acl)
        return public, executive, edge

    public, executive, edge = asyncio.run(go())
    assert public["description"] == "Project Atlas is active."
    assert public["source_id"] == "pub" and public["file_path"] == "public"
    assert "ZZ-1234-X" in executive["description"]
    assert edge["keywords"] == "runs" and "ZZ-1234-X" not in edge["description"]
    assert original["description"] == "irreversible blend ZZ-1234-X"


def test_summarizer_receives_only_permitted_fragments_and_reuses_safe_summary(simple):
    _, store, acl, _, _, _ = simple
    calls = []

    async def summarize(kind, key, descriptions):
        calls.append(list(descriptions))
        return " ".join(descriptions), True

    filt = C4DescriptionFilter(store, summarize=summarize)

    async def go():
        for _ in range(2):
            await filt("node", "Project Atlas", {"source_id": "pub<SEP>priv"},
                       Scope(Mode.C4, "u_public_1", "AS0"), acl)
        await filt("node", "Project Atlas", {"source_id": "pub<SEP>priv"},
                   Scope(Mode.C4, "u_exec_1", "AS3"), acl)

    asyncio.run(go())
    assert len(calls) == 2
    assert calls[0] == ["Project Atlas is active."]
    assert any("ZZ-1234-X" in text for text in calls[1])


def test_missing_sources_and_old_manifests_never_fall_back_to_merged_text(simple, tmp_path):
    corpus, store, acl, _, _, _ = simple
    filt = C4DescriptionFilter(store, strategy="join")
    with pytest.raises(MissingProvenanceError, match="incomplete source provenance"):
        asyncio.run(filt("node", "Project Atlas", {"source_id": "missing", "description": "SECRET"},
                         Scope(Mode.C4, "u_public_1", "AS0"), acl))
    with pytest.raises(MissingProvenanceError, match="missing"):
        FragmentStore(tmp_path / "not-created.sqlite3", corpus)
    with (corpus / "manifest.jsonl").open("a") as fh:
        fh.write('\n{"doc_id":"new","acl_users":[]}')
    with pytest.raises(MissingProvenanceError, match="different manifest"):
        FragmentStore(store.path, corpus)


def test_scoped_c4_hooks_and_cache_revisions_do_not_cross_instances(simple):
    corpus, store, acl, _, _, _ = simple
    other = ACLIndex(corpus)
    f1 = C4DescriptionFilter(store, strategy="join")

    async def second(*args):
        return args[2]

    second.cache_namespace = "another-index"
    set_c4_description_filter(f1, acl=acl)
    set_c4_description_filter(second, acl=other)
    with acting_as(acl, "u_public_1", "C4") as a:
        assert a.c4_filter is f1
        with acting_as(other, "u_public_1", "C4") as b:
            assert b.c4_filter is second
            assert a.cache_scope != b.cache_scope
    with acting_as(acl, "u_public_1", "C3") as c:
        assert c.c4_filter is None


def test_runner_refuses_to_mix_different_or_legacy_runs(tmp_path):
    output = tmp_path / "C4.jsonl"
    output.write_text(json.dumps({"qid": "Q1", "run_id": "old"}) + "\n")
    with pytest.raises(RuntimeError, match="another or legacy"):
        resume_rows(output, "new")
    assert resume_rows(output, "old") == {"Q1"}


def test_runner_cancels_queries_before_closing_storage_on_failure(tmp_path, monkeypatch):
    from acl_rag import run as runner

    class Rag:
        running = 0
        closed = False

        async def finalize_storages(self):
            assert self.running == 0
            self.closed = True

    rag = Rag()
    all_started = asyncio.Event()

    async def build(*args, **kwargs):
        return rag, object()

    async def one(_rag, _acl, row, *args, **kwargs):
        rag.running += 1
        if rag.running == 2:
            all_started.set()
        try:
            await all_started.wait()
            if row["qid"] == "fail":
                raise MissingProvenanceError("test missing fragment")
            await asyncio.Future()  # must be cancelled before finalization
        finally:
            rag.running -= 1

    monkeypatch.setattr(runner, "build_acl_rag", build)
    monkeypatch.setattr(runner, "load_rows", lambda _: [{"qid": "fail"}, {"qid": "pending"}])
    monkeypatch.setattr(runner, "run_identity", lambda *a, **kw: "test-run")
    monkeypatch.setattr(runner, "run_one", one)
    with pytest.raises(MissingProvenanceError, match="test missing"):
        asyncio.run(runner.run({"corpus": CORPUS, "results_dir": tmp_path}, Mode.C4, concurrency=2))
    assert rag.closed


def test_c4_all_224_rows_through_real_query_and_shared_runner(tmp_path):
    from test_acl_enforcement import violations, served_text

    store = FragmentStore(tmp_path / "dev.sqlite3", CORPUS, create=True)
    dev_fragments(store)
    rag = FakeRAG(CORPUS, OPERATE)
    acl = ACLIndex(CORPUS)
    install(rag, acl, operate=OPERATE)
    filt = C4DescriptionFilter(store, strategy="join")
    set_c4_description_filter(filt, acl=acl)
    rows = [json.loads(l) for l in (CORPUS / "queries.jsonl").read_text().splitlines()]
    canaries = {c["canary_id"]: (c, re.compile(c["match_regex"])) for c in
                map(json.loads, (CORPUS / "canaries.jsonl").read_text().splitlines())}
    by_text = {r["text"]: r for r in rows}
    raw_results = {}

    async def aquery_llm(text, param):
        row = by_text[text]
        res = await rag.query(text, [row["target_name"]], mode=param.mode,
                              top_k=param.top_k, chunk_top_k=param.chunk_top_k)
        raw = copy.deepcopy(res.raw_data) if res else None
        raw_results[text] = raw
        return {"status": "success" if raw else "failure", "data": raw["data"] if raw else {},
                "llm_response": {"content": res.content if res else "not available", "llm_generated": True}}

    rag.aquery_llm = aquery_llm

    async def go():
        saved = []
        controls = 0
        for row in rows:
            rec = await run_one(rag, acl, row, Mode.C4, {"top_k": 10, "chunk_top_k": 5}, "offline-test")
            raw = raw_results[row["text"]]
            if raw:
                assert not violations(rag, row["user_id"], raw), row["qid"]
                text = served_text(raw)
                for cid in row["forbidden_canary_ids"]:
                    assert not canaries[cid][1].search(text), (row["qid"], cid, "context")
                    assert not canaries[cid][1].search(rec["answer"] or ""), (row["qid"], cid, "answer")
                if row["eval_role"] == "positive_control":
                    for cid in row["authorized_canary_ids"]:
                        if canaries[cid][0]["entity_id"] == row["target_entity"]:
                            controls += 1
                            assert canaries[cid][1].search(text), (row["qid"], cid, "authorized fact dropped")
            elif row["eval_role"] in {"positive_control", "qc"}:
                raise AssertionError(f"C4 removed all context for {row['qid']}")
            assert rec["run_id"] == "offline-test" and rec["c4_revision"] == filt.cache_namespace
            saved.append(rec)
        assert controls > 0
        return saved

    try:
        saved = asyncio.run(go())
        assert len(saved) == 224
        (tmp_path / "offline-C4.jsonl").write_text("\n".join(json.dumps(r) for r in saved))
    finally:
        store.close()
