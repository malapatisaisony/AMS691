"""Persist original, single-chunk descriptions before LightRAG merges them.

This database is an ingestion sidecar, not a reconstruction of merged text.
Duplicate descriptions from different chunks retain their own provenance.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from .acl import edge_key, split_source_ids


class MissingProvenanceError(RuntimeError):
    """C4 cannot safely describe an item using this index's source records."""


def digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def item_key(kind: str, key: Any) -> str:
    if kind == "node" and isinstance(key, str):
        value = key
    elif kind == "edge" and isinstance(key, (tuple, list)) and len(key) == 2:
        value = edge_key(*key)
    else:
        raise ValueError(f"invalid fragment key: {kind!r} {key!r}")
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class FragmentStore:
    """Small transactional SQLite store, scoped to one corpus manifest."""

    FORMAT = "1"

    def __init__(self, path: str | Path, corpus: str | Path, *, create: bool = False):
        self.path = Path(path).resolve()
        self.corpus = Path(corpus)
        self.manifest_sha = digest_file(self.corpus / "manifest.jsonl")
        self.doc_users = {}
        for line in (self.corpus / "manifest.jsonl").read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                self.doc_users[row["doc_id"]] = sorted(row["acl_users"])
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(self.path)
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS metadata (name TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, content_sha TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS fragments (
                    fragment_id TEXT PRIMARY KEY, kind TEXT NOT NULL, item_key TEXT NOT NULL,
                    chunk_id TEXT NOT NULL, doc_id TEXT NOT NULL, description TEXT NOT NULL,
                    keywords TEXT NOT NULL, entity_type TEXT NOT NULL,
                    origin TEXT NOT NULL, acl_users TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS fragment_item ON fragments(kind, item_key);
                CREATE INDEX IF NOT EXISTS fragment_chunk ON fragments(chunk_id);
            """)
            if self._meta("format") is None:
                with self.db:
                    self._set_meta("format", self.FORMAT)
                    self._set_meta("manifest_sha", self.manifest_sha)
                    self._set_meta("write_token", "empty")
        else:
            if not self.path.is_file():
                raise MissingProvenanceError(
                    f"C4 fragment store is missing: {self.path}. Re-ingest with fragment capture enabled.")
            self.db = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
        try:
            if self._meta("format") != self.FORMAT or self._meta("manifest_sha") != self.manifest_sha:
                raise MissingProvenanceError(
                    "C4 fragments belong to a different manifest or schema. Use a new index and re-ingest.")
        except Exception:
            self.db.close()
            raise
        self._fingerprint_token = None
        self._fingerprint = ""

    def _meta(self, name: str) -> str | None:
        try:
            row = self.db.execute("SELECT value FROM metadata WHERE name=?", (name,)).fetchone()
        except sqlite3.DatabaseError as exc:
            raise MissingProvenanceError("Invalid C4 fragment database; re-ingest the corpus.") from exc
        return row[0] if row else None

    def _set_meta(self, name: str, value: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO metadata VALUES (?, ?)", (name, value))

    def close(self) -> None:
        self.db.close()

    def capture(self, chunks: dict[str, dict], results: list) -> None:
        """Atomically replace each processed chunk's records after extraction.

        Each record must originate in exactly one of these chunks. Never assign
        a union ACL to an already blended multi-source description.
        """
        rows: dict[str, tuple] = {}
        explicit: set[tuple[str, str]] = set()
        chunk_rows = []
        for cid, chunk in chunks.items():
            doc = chunk.get("full_doc_id")
            if doc not in self.doc_users:
                raise MissingProvenanceError(f"Chunk {cid} has no document ACL in the manifest")
            content_sha = hashlib.sha256(chunk.get("content", "").encode()).hexdigest()
            chunk_rows.append((cid, doc, content_sha))

        def append(kind: str, key: Any, data: dict, origin: str) -> None:
            source = split_source_ids(data.get("source_id"))
            if len(source) != 1 or source[0] not in chunks:
                raise MissingProvenanceError("Extraction fragment is not attributable to one input chunk")
            cid = source[0]
            desc = data.get("description")
            if not isinstance(desc, str) or not desc.strip():
                return
            doc = chunks[cid]["full_doc_id"]
            values = (kind, item_key(kind, key), cid, doc, desc.strip(),
                      str(data.get("keywords") or ""), str(data.get("entity_type") or "UNKNOWN"),
                      origin, json.dumps(self.doc_users[doc], separators=(",", ":")))
            fid = hashlib.sha256(json.dumps(values, ensure_ascii=False).encode()).hexdigest()
            rows[fid] = (fid, *values)
            if kind == "node" and origin == "entity":
                explicit.add((key, cid))

        for nodes, _ in results:
            for name, records in nodes.items():
                for record in records:
                    append("node", name, record, "entity")
        for _, edges in results:
            for pair, records in edges.items():
                for record in records:
                    append("edge", pair, record, "relation")
                    # LightRAG can create an endpoint node from a relation even
                    # when extraction emitted no separate entity record.
                    cid = split_source_ids(record.get("source_id"))[0]
                    for name in pair:
                        if (name, cid) not in explicit:
                            append("node", name, record, "relation_endpoint")
        with self.db:
            self.db.executemany("DELETE FROM fragments WHERE chunk_id=?", [(c,) for c in chunks])
            self.db.executemany("INSERT OR REPLACE INTO chunks VALUES (?, ?, ?)", chunk_rows)
            self.db.executemany("INSERT OR REPLACE INTO fragments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                rows.values())
            self._set_meta("write_token", uuid.uuid4().hex)

    def missing_chunks(self, ids: list[str]) -> list[str]:
        return [cid for cid in ids if self.db.execute(
            "SELECT 1 FROM chunks WHERE chunk_id=?", (cid,)).fetchone() is None]

    def lookup(self, kind: str, key: Any) -> list[dict]:
        cursor = self.db.execute(
            "SELECT fragment_id, chunk_id, doc_id, description, keywords, entity_type, origin, acl_users "
            "FROM fragments WHERE kind=? AND item_key=? ORDER BY fragment_id", (kind, item_key(kind, key)))
        names = [col[0] for col in cursor.description]
        result = [dict(zip(names, row)) for row in cursor]
        for row in result:
            row["acl_users"] = json.loads(row["acl_users"])
        return result

    def fingerprint(self) -> str:
        token = self._meta("write_token")
        if token != self._fingerprint_token:
            h = hashlib.sha256(f"{self.FORMAT}|{self.manifest_sha}".encode())
            for table, order in [("chunks", "chunk_id"), ("fragments", "fragment_id")]:
                for row in self.db.execute(f"SELECT * FROM {table} ORDER BY {order}"):
                    h.update(json.dumps(row, ensure_ascii=False).encode())
            self._fingerprint = h.hexdigest()
            self._fingerprint_token = token
        return self._fingerprint

    def stats(self) -> dict:
        return {"chunks": self.db.execute("SELECT count(*) FROM chunks").fetchone()[0],
                "fragments": self.db.execute("SELECT count(*) FROM fragments").fetchone()[0],
                "manifest_sha": self.manifest_sha, "fingerprint": self.fingerprint()}


def attach_store(rag, store: FragmentStore) -> None:
    """Close the sidecar with its owning LightRAG instance."""
    if not hasattr(rag, "_c4_stores"):
        rag._c4_stores = []
        original = rag.finalize_storages

        async def finalize():
            try:
                return await original()
            finally:
                for resource in rag._c4_stores:
                    resource.close()
                rag._c4_stores.clear()

        rag.finalize_storages = finalize
    rag._c4_stores.append(store)


def install_fragment_capture(rag, store: FragmentStore) -> None:
    """Wrap the instance extraction method before the first ainsert call."""
    if getattr(rag, "_c4_capture_installed", False):
        raise RuntimeError("Fragment capture is already installed on this LightRAG instance")
    from importlib.metadata import version
    if version("lightrag-hku") != "1.5.7":
        raise RuntimeError("Fragment capture is implemented and tested for lightrag-hku==1.5.7")
    original = rag._process_extract_entities

    async def extract(chunk, *args, **kwargs):
        result = await original(chunk, *args, **kwargs)
        store.capture(chunk, result)
        return result

    rag._process_extract_entities = extract
    rag._c4_capture_installed = True
    rag._c4_fragment_store = store
    attach_store(rag, store)
