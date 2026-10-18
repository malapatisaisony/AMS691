"""C4: compose descriptions from authorized original source fragments only."""
from __future__ import annotations

import asyncio
import hashlib
import json
from collections import Counter
from typing import Any, Awaitable, Callable

from .acl import ACLIndex, GRAPH_FIELD_SEP, split_source_ids
from .fragments import FragmentStore, MissingProvenanceError, digest_file, item_key

Summary = Callable[[str, Any, list[str]], Awaitable[tuple[str, bool]]]


class C4DescriptionFilter:
    """The async callable expected by set_c4_description_filter.

    ``auto`` uses LightRAG's summarizer on permitted fragments; ``join`` is a
    deterministic alternative for offline tests. Neither reads merged text.
    """

    POLICY = "source-fragments-v1"

    def __init__(self, store: FragmentStore, *, strategy: str = "auto", summarize: Summary | None = None):
        if strategy not in {"auto", "join"}:
            raise ValueError("c4.description_strategy must be auto or join")
        if strategy == "auto" and summarize is None:
            raise ValueError("auto descriptions require an authorized-fragment summarizer")
        self.store, self.strategy, self.summarize = store, strategy, summarize
        self.enable_cache = True
        self._summary_identity = getattr(summarize, "cache_identity", f"custom-{id(summarize)}") if summarize else "none"
        self._cache: dict[str, tuple[str, bool]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._oracle_sha = digest_file(store.corpus / "oracle_splits.json")

    @property
    def cache_namespace(self) -> str:
        identity = (
            f"{self.POLICY}|{self.strategy}|{self.enable_cache}|{self.store.fingerprint()}|{self._oracle_sha}|"
            f"{self._summary_identity}")
        return hashlib.sha256(identity.encode()).hexdigest()

    async def __call__(self, kind: str, key: Any, data: dict, scope, acl: ACLIndex) -> dict:
        if digest_file(acl.corpus_dir / "manifest.jsonl") != self.store.manifest_sha:
            raise MissingProvenanceError("C4 filter and ACL index use different corpus permissions")
        ids = split_source_ids(data.get("source_id"))
        if not ids or self.store.missing_chunks(ids):
            raise MissingProvenanceError(
                f"C4 has incomplete source provenance for {kind} {key!r}. Re-ingest with capture enabled.")
        acls = await acl.chunk_acls(ids)
        docs = await acl.chunk_docs(ids)
        allowed = set(ids)
        fragments = [f for f in self.store.lookup(kind, key)
                     if f["chunk_id"] in allowed and f["doc_id"] == docs.get(f["chunk_id"])
                     and scope.user_id in acls.get(f["chunk_id"], ())
                     and scope.user_id in f["acl_users"]]
        if not fragments:
            raise MissingProvenanceError(
                f"C4 has no authorized original descriptions for {kind} {key!r}; merged text is never a fallback")

        descriptions = list(dict.fromkeys(f["description"] for f in fragments))
        sig = hashlib.sha256(json.dumps(
            [self.cache_namespace, scope.access_set, kind, item_key(kind, key),
             [f["fragment_id"] for f in fragments]], ensure_ascii=False).encode()).hexdigest()
        async def compose():
            if self.strategy == "join":
                value = GRAPH_FIELD_SEP.join(descriptions), False
            else:
                value = await self.summarize(kind, key, descriptions)
            if not isinstance(value[0], str) or not value[0].strip():
                raise RuntimeError("C4 summarizer returned an empty description")
            return value

        if not self.enable_cache:
            description, used_llm = await compose()
        else:
            async with self._locks.setdefault(sig, asyncio.Lock()):
                if sig not in self._cache:
                    self._cache[sig] = await compose()
                description, used_llm = self._cache[sig]
        used = {f["chunk_id"] for f in fragments}
        keep_ids = [cid for cid in ids if cid in used]
        out = {**data, "description": description, "source_id": GRAPH_FIELD_SEP.join(keep_ids),
               "file_path": GRAPH_FIELD_SEP.join(dict.fromkeys(docs[cid] for cid in keep_ids)),
               "c4_fragment_ids": [f["fragment_id"] for f in fragments], "c4_summary_used": used_llm}
        if kind == "edge":
            # Edge keywords can also contain source facts: never keep the
            # globally merged keyword field when descriptions are filtered.
            out["keywords"] = ", ".join(dict.fromkeys(f["keywords"] for f in fragments if f["keywords"]))
        else:
            types = [f["entity_type"] for f in fragments if f["entity_type"] != "UNKNOWN"]
            out["entity_type"] = Counter(types).most_common(1)[0][0] if types else "UNKNOWN"
        return out


def lightrag_summarizer(rag) -> Summary:
    """Apply the pinned merge policy, with only authorized inputs."""
    from lightrag.operate import _handle_entity_relation_summary
    config = rag._build_global_config()

    async def summarize(kind: str, key: Any, descriptions: list[str]) -> tuple[str, bool]:
        return await _handle_entity_relation_summary(
            "Entity" if kind == "node" else "Relation", str(key), descriptions,
            GRAPH_FIELD_SEP, config, rag.llm_response_cache if rag.enable_llm_cache else None)

    summarize.cache_identity = hashlib.sha256(json.dumps(
        {k: config.get(k) for k in ("llm_model_name", "force_llm_summary_on_merge",
         "summary_max_tokens", "summary_context_size", "summary_length_recommended",
         "llm_cache_identities")}, sort_keys=True, default=str).encode()).hexdigest()
    return summarize
