"""Real LightRAG ingestion, persistence, factory wiring, and query regression.

The LLM and embedding functions are deterministic local stubs. Extraction
parsing, chunk/graph/vector stores, merging, cache, and queries are real 1.5.7.
No network calls, API keys, or provider charges are required.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from acl_rag import Mode, acting_as
from acl_rag.fragments import FragmentStore
from acl_rag.ingest import ingest
from acl_rag import rag_factory


class CharacterCodec:
    def encode(self, text):
        return list(map(ord, text))

    def decode(self, tokens):
        return "".join(map(chr, tokens))


def test_real_ingestion_reopen_and_c4_summaries(tmp_path, monkeypatch):
    import lightrag
    from lightrag import QueryParam
    from lightrag.utils import EmbeddingFunc, Tokenizer

    corpus = tmp_path / "corpus"
    (corpus / "docs").mkdir(parents=True)
    public, executive = "u_public_1", "u_exec_1"
    source_facts = {
        "PUBLIC_ONE": "Project Atlas is active under Team North.",
        "PUBLIC_TWO": "Project Atlas has a public launch in April with Team North.",
        "RESTRICTED_ONE": "Project Atlas and Team North use restricted budget code ZZ-1234-X.",
    }
    docs = []
    for marker, fact in source_facts.items():
        doc = {"doc_id": marker, "acl_users": [executive] if marker.startswith("RESTRICTED")
               else [public, executive]}
        docs.append(doc)
        (corpus / "docs" / f"{marker}.txt").write_text(f"{marker}\n{fact}")
    (corpus / "manifest.jsonl").write_text("\n".join(json.dumps(d) for d in docs))
    (corpus / "oracle_splits.json").write_text(json.dumps({
        "access_sets": {"AS0": {}, "AS3": {}},
        "user_to_access_set": {public: "AS0", executive: "AS3"}}))
    (corpus / "facts.json").write_text(json.dumps({"entities": [
        {"entity_id": "E1", "canonical_name": "Project Atlas", "cls": "cross", "hub": True},
        {"entity_id": "E2", "canonical_name": "Team North", "cls": "cross", "hub": True}]}))

    summary_calls = []

    async def llm(prompt, system_prompt=None, **kwargs):
        # A summary prompt contains JSONL originals; echo exactly those inputs.
        if "Description List:" in prompt:
            inputs = [json.loads(line)["Description"] for line in prompt.splitlines()
                      if line.startswith('{"Description":')]
            assert inputs
            summary_calls.append(inputs)
            return " ".join(dict.fromkeys(inputs))
        if "expert keyword extractor" in prompt:
            return json.dumps({"high_level_keywords": ["project"],
                               "low_level_keywords": ["Project Atlas", "Team North"]})
        for marker, fact in source_facts.items():
            if marker in prompt:
                return json.dumps({"entities": [
                    {"name": "Project Atlas", "type": "PROJECT", "description": fact},
                    {"name": "Team North", "type": "ORGANIZATION", "description": fact}],
                    "relationships": [{"source": "Project Atlas", "target": "Team North",
                                       "description": fact,
                                       "keywords": "ZZ-1234-X" if "ZZ-1234-X" in fact else "project"}]})
        if system_prompt and "Knowledge Graph Data" in system_prompt:
            return "Budget code ZZ-1234-X." if "ZZ-1234-X" in system_prompt else "Project Atlas is active."
        raise AssertionError(f"Unexpected model call: {prompt[:100]}")

    async def embed(texts, **kwargs):
        # Identical unit vectors make every test record retrievable; the ACL
        # and C4 layers must remove restricted material, not a lucky ranking.
        return np.ones((len(texts), 8), dtype=np.float32) / np.sqrt(8)

    monkeypatch.setattr(rag_factory, "_llm_and_embedding", lambda _: (
        llm, EmbeddingFunc(embedding_dim=8, max_token_size=8192, func=embed)))
    real_rag = lightrag.LightRAG

    def local_rag(**kwargs):
        return real_rag(**kwargs, tokenizer=Tokenizer("test-characters", CharacterCodec()),
                        entity_extraction_use_json=True, entity_extract_max_gleaning=0)

    monkeypatch.setattr(lightrag, "LightRAG", local_rag)
    cfg = {"corpus": corpus, "working_dir": tmp_path / "index",
           "results_dir": tmp_path / "results", "backend": {"llm": "test", "llm_model": "stub"},
           "lightrag": {"chunk_token_size": 1200, "chunk_overlap_token_size": 0,
                        "force_llm_summary_on_merge": 3, "summary_max_tokens": 80,
                        "summary_length_recommended": 60},
           "c4": {"description_strategy": "auto"}}

    async def go():
        report = await ingest(cfg)
        assert report["failed_docs"] == [] and report["chunks_with_wrong_parent"] == []
        assert report["n_chunks"] == 3 and report["entities_found"] == 2
        assert report["c4_provenance"]["fragments"] == 9
        with_store = FragmentStore(rag_factory.fragment_path(cfg), corpus)
        try:
            originals = with_store.lookup("node", "Project Atlas")
            assert {r["description"] for r in originals} == set(source_facts.values())
            assert len({r["chunk_id"] for r in originals}) == 3
            for row in originals:
                assert row["description"] == source_facts[row["doc_id"]]
        finally:
            with_store.close()

        # Factory must reopen the persisted sidecar and register C4 itself.
        rag, acl = await rag_factory.build_acl_rag(cfg, mode=Mode.C4)
        try:
            node = await rag.chunk_entity_relation_graph.get_node("Project Atlas")
            assert "ZZ-1234-X" in node["description"]  # contaminated global summary
            summary_calls.clear()
            callback = acl.c4_description_filter
            with acting_as(acl, public, "C4") as scope:
                safe = await callback("node", "Project Atlas", node, scope, acl)
            assert safe["c4_summary_used"] and "ZZ-1234-X" not in safe["description"]
            assert summary_calls and all("ZZ-1234-X" not in d for call in summary_calls for d in call)
            assert all(source_facts[m] in safe["description"] for m in ("PUBLIC_ONE", "PUBLIC_TWO"))
            count = len(summary_calls)
            with acting_as(acl, public, "C4") as scope:
                await callback("node", "Project Atlas", node, scope, acl)
            assert len(summary_calls) == count

            # Exercise actual aquery_llm, vector stores, one-hop graph reads,
            # chunk filtering, and access-set answer caches after reopening.
            for mode, user, should_contain_secret in [
                ("C3", public, True), ("C4", public, False),
                ("C4", executive, True), ("C4", public, False),
            ]:
                with acting_as(acl, user, mode):
                    result = await rag.aquery_llm("Describe Project Atlas and Team North.",
                        param=QueryParam(mode="mix", top_k=5, chunk_top_k=5,
                                         enable_rerank=False, max_total_tokens=30000))
                assert result["status"] == "success", result
                text = json.dumps(result["data"]) + result["llm_response"]["content"]
                assert ("ZZ-1234-X" in text) == should_contain_secret, (mode, user, text)
                if user == public and mode == "C4":
                    assert "RESTRICTED_ONE" not in json.dumps(result["data"])
            unchanged = await rag.chunk_entity_relation_graph.get_node("Project Atlas")
            assert unchanged == node

            # No-cache timing runs must bypass both the in-memory C4 cache
            # and LightRAG's persisted summary cache, even for repeated inputs.
            rag.enable_llm_cache = False
            callback.enable_cache = False
            count = len(summary_calls)
            for _ in range(2):
                with acting_as(acl, public, "C4") as scope:
                    await callback("node", "Project Atlas", node, scope, acl)
            assert len(summary_calls) == count + 2
        finally:
            await rag.finalize_storages()

    asyncio.run(go())
