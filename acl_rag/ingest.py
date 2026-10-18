"""Ingest a corpus build into stock LightRAG (C1 index; C2-C4 query the same index).

    python -m acl_rag.ingest --rag-config configs/rag.yaml

Each document goes in with ``ainsert(text, ids=doc_id, file_paths=doc_id)`` so
every chunk records its parent doc (``full_doc_id``) and every retrieved chunk
carries the doc id as its ``file_path``.

Writes into the index directory:
  chunk_map.json      doc_id -> [chunk ids]   (for origin_chunks / ACL checks)
  ingest_report.json  per-entity merge check: does a node exist under the exact
                      canonical name, how many source chunks it has, and the
                      source-count hub proxy (>= 8 chunks vs < 8).
  c4_fragments.sqlite3 original entity/relation descriptions and source ACLs,
                      captured before merging (unless disabled in config).

The source-count proxy is not an observation of whether an LLM summary ran;
LightRAG also applies token limits and may merge prior summaries incrementally.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from .acl import split_source_ids
from .rag_factory import build_rag, load_config

log = logging.getLogger("acl_rag.ingest")


def read_docs(corpus: Path) -> tuple[list[str], list[str]]:
    ids, texts = [], []
    with open(corpus / "manifest.jsonl") as fh:
        for line in fh:
            if line.strip():
                doc_id = json.loads(line)["doc_id"]
                ids.append(doc_id)
                texts.append((corpus / "docs" / f"{doc_id}.txt").read_text())
    return ids, texts


async def ingest(cfg: dict) -> dict:
    corpus: Path = cfg["corpus"]
    ids, texts = read_docs(corpus)
    rag = await build_rag(cfg, capture_fragments=True)
    try:
        log.info("inserting %d docs from %s into %s", len(ids), corpus, cfg["working_dir"])
        await rag.ainsert(texts, ids=ids, file_paths=ids)

        # ---- chunk map + doc status
        chunk_map, failed = {}, []
        for doc_id in ids:
            st = await rag.doc_status.get_by_id(doc_id) or {}
            status = str(st.get("status", "")).lower()
            if "processed" not in status:
                failed.append({"doc_id": doc_id, "status": status, "error": st.get("error_msg")})
            chunk_map[doc_id] = list(st.get("chunks_list") or [])

        # every chunk must point back to its own doc
        bad_parent = []
        for doc_id, cids in chunk_map.items():
            rows = await rag.text_chunks.get_by_ids(cids) if cids else []
            for cid, row in zip(cids, rows):
                if (row or {}).get("full_doc_id") != doc_id:
                    bad_parent.append(cid)

        # ---- merge check on our canonical names
        facts = json.loads((corpus / "facts.json").read_text())
        graph = rag.chunk_entity_relation_graph
        threshold = cfg.get("lightrag", {}).get("force_llm_summary_on_merge", 8)
        entities, missing = [], []
        for e in facts["entities"]:
            node = await graph.get_node(e["canonical_name"])
            n_src = len(split_source_ids((node or {}).get("source_id")))
            row = {"entity_id": e["entity_id"], "name": e["canonical_name"], "cls": e["cls"],
                   "planned_hub": e.get("hub"), "found": node is not None,
                   "n_source_chunks": n_src,
                   "realized_regime": "hub" if n_src >= threshold else "non_hub"}
            entities.append(row)
            if node is None:
                missing.append(e["canonical_name"])

        report = {
            "corpus": str(corpus), "n_docs": len(ids), "failed_docs": failed,
            "n_chunks": sum(len(v) for v in chunk_map.values()),
            "chunks_with_wrong_parent": bad_parent,
            "entities_found": sum(r["found"] for r in entities), "entities_total": len(entities),
            "entities_missing": missing,
            "realized_hubs": sum(r["realized_regime"] == "hub" for r in entities),
            "hub_prediction_mismatches": [r["name"] for r in entities if r["found"] and
                                          (r["realized_regime"] == "hub") != bool(r["planned_hub"])],
            "entities": entities,
            "c4_provenance": rag._c4_fragment_store.stats()
                if hasattr(rag, "_c4_fragment_store") else None,
        }
        wd: Path = cfg["working_dir"]
        (wd / "chunk_map.json").write_text(json.dumps(chunk_map, indent=1))
        (wd / "ingest_report.json").write_text(json.dumps(report, indent=1))
        return report
    finally:
        await rag.finalize_storages()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rag-config", default="configs/rag.yaml")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    r = asyncio.run(ingest(load_config(args.rag_config)))
    print(json.dumps({k: v for k, v in r.items() if k != "entities"}, indent=1))
    if r["failed_docs"] or r["chunks_with_wrong_parent"]:
        raise SystemExit("ingest finished with errors (see above)")


if __name__ == "__main__":
    main()
