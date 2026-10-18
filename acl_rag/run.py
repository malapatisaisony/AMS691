"""Run every (query, user) row of queries.jsonl under one configuration.

    python -m acl_rag.run --rag-config configs/rag.yaml --config C2
    python -m acl_rag.run --rag-config configs/rag.yaml --config C3 --limit 16

Output: <results_dir>/<config>.jsonl, one record per (query, user), resumable
(rows already in the file are skipped). Each record has what was retrieved
(entities and relations with their served descriptions and source_ids, chunks
with their doc ids, in rank order), the answer, and latency. Leakage /
contamination are computed from these files by the metrics owner; recall@k,
answer quality and latency by ``acl_rag.metrics``.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path

from .enforce import Mode, acting_as
from .rag_factory import build_acl_rag, load_config

log = logging.getLogger("acl_rag.run")


def load_rows(corpus: Path) -> list[dict]:
    with open(corpus / "queries.jsonl") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def _query_param(qcfg: dict):
    from lightrag import QueryParam  # noqa: PLC0415
    return QueryParam(  # fresh object per call; never share a QueryParam
        mode=qcfg.get("mode", "mix"),
        top_k=qcfg.get("top_k", 40),
        chunk_top_k=qcfg.get("chunk_top_k", 20),
        enable_rerank=qcfg.get("enable_rerank", False),
        response_type=qcfg.get("response_type", "Multiple Paragraphs"),
    )


async def run_one(rag, acl, row: dict, mode: Mode, qcfg: dict, run_id: str | None = None) -> dict:
    t0 = time.perf_counter()
    with acting_as(acl, row["user_id"], mode) as scope:
        res = await rag.aquery_llm(row["text"], param=_query_param(qcfg))
    latency = time.perf_counter() - t0
    data = res.get("data") or {}
    llm = res.get("llm_response") or {}
    chunks = data.get("chunks") or []
    return {
        "qid": row["qid"], "query_id": row["query_id"], "type": row["type"],
        "user_id": row["user_id"], "user_tier": row["user_tier"],
        "access_set": scope.access_set, "config": mode.value, "eval_role": row["eval_role"],
        "run_id": run_id, "c4_revision": scope.c4_revision if mode is Mode.C4 else None,
        "status": res.get("status"),
        "answer": llm.get("content"),
        "llm_generated": llm.get("llm_generated"),
        "latency_s": round(latency, 4),
        "retrieved": {
            "entities": [{"name": e.get("entity_name"), "description": e.get("description"),
                          "source_id": e.get("source_id")} for e in data.get("entities") or []],
            "relations": [{"src": r.get("src_id"), "tgt": r.get("tgt_id"),
                           "description": r.get("description"), "source_id": r.get("source_id")}
                          for r in data.get("relationships") or []],
            "chunks": [{"rank": i + 1, "chunk_id": c.get("chunk_id"), "doc_id": c.get("file_path"),
                        "content": c.get("content")} for i, c in enumerate(chunks)],
        },
        "keywords": (res.get("metadata") or {}).get("keywords"),
    }


def run_identity(cfg: dict, acl, mode: Mode, *, concurrency: int, no_llm_cache: bool) -> str:
    """Keep incompatible experiments out of the same resumable result file."""
    inputs = {}
    for name in ("manifest.jsonl", "queries.jsonl", "facts.json", "canaries.jsonl", "oracle_splits.json"):
        inputs[name] = hashlib.sha256((cfg["corpus"] / name).read_bytes()).hexdigest()
    callback = getattr(acl, "c4_description_filter", None)
    backend = cfg.get("backend", {})
    signature = {"schema": 1, "config": mode.value, "inputs": inputs,
                 "query": cfg.get("query", {}), "lightrag": cfg.get("lightrag", {}),
                 "acl": cfg.get("acl", {}), "backend": {k: backend.get(k) for k in (
                     "llm", "llm_model", "embedding", "embedding_model", "embedding_dim")},
                 "concurrency": concurrency, "no_llm_cache": no_llm_cache,
                 "c4_revision": getattr(callback, "cache_namespace", None) if mode is Mode.C4 else None}
    return hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()


def resume_rows(path: Path, run_id: str) -> set[str]:
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("run_id") != run_id:
                raise RuntimeError(
                    f"{path} belongs to another or legacy experiment. Use a fresh results_dir; "
                    "existing results will not be mixed or overwritten.")
            done.add(row["qid"])
    return done


async def run(cfg: dict, mode: Mode, limit: int | None = None, concurrency: int = 1,
              no_llm_cache: bool = False) -> Path:
    rows = load_rows(cfg["corpus"])
    if limit:
        rows = rows[:limit]
    out_dir: Path = cfg["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{mode.value}.jsonl"
    rag, acl = await build_acl_rag(cfg, mode=mode)
    if no_llm_cache:
        rag.enable_llm_cache = False
        callback = getattr(acl, "c4_description_filter", None)
        if mode is Mode.C4 and callback is not None:
            callback.enable_cache = False
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    try:
        run_id = run_identity(cfg, acl, mode, concurrency=concurrency, no_llm_cache=no_llm_cache)
        done = resume_rows(out, run_id)
        todo = [r for r in rows if r["qid"] not in done]
        log.info("%s: %d rows (%d already done)", mode.value, len(todo), len(done))
        with open(out, "a") as fh:
            async def one(row):
                async with sem:
                    rec = await run_one(rag, acl, row, mode, cfg.get("query", {}), run_id=run_id)
                async with lock:
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
            tasks = [asyncio.create_task(one(r)) for r in todo]
            try:
                await asyncio.gather(*tasks)
            finally:
                # A missing C4 source must stop the run before storages and
                # the result file close underneath other in-flight queries.
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        await rag.finalize_storages()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rag-config", default="configs/rag.yaml")
    ap.add_argument("--config", required=True, help="C1 | C2 | C3 | C4")
    ap.add_argument("--limit", type=int, help="only the first N rows (smoke test)")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="parallel queries; keep 1 when latency numbers matter")
    ap.add_argument("--no-llm-cache", action="store_true",
                    help="bypass query-answer and C4 summary caches (use for latency runs)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    out = asyncio.run(run(load_config(args.rag_config), Mode.parse(args.config), args.limit,
                          args.concurrency, args.no_llm_cache))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
