"""ACL enforcement for LightRAG 1.5.7: C1 / C2 / C3 (+ a C4 hook) behind one switch.

One LightRAG instance serves every configuration and every user. The active
configuration and user live in a context variable, set per query with
:func:`acting_as`; the patches below read it and do nothing when it is unset
(so ingestion and any non-ACL use run exactly as stock LightRAG).

  C1  stock retrieval, no ACL checks.
  C2  post-retrieval dropout: LightRAG retrieves as usual (vector search and
      one-hop expansion), then every entity, relation and chunk with
      u not in ACL(v) is dropped before the context is built.
      Hooks: operate._perform_kg_search, operate._merge_all_chunks.
  C3  pre-filtering: the same check is applied *inside* retrieval, so an
      unauthorized item is never a candidate:
        - entity / relation / chunk vector stores (oversampled, then filtered,
          so top_k still means top_k authorized items),
        - one-hop expansion (graph reads of nodes, a node's edges, edges),
        - chunk reads (text store and chunk vectors), and each node's or edge's
          source_id is cut down to the chunks the user may read.
      A node's merged *description* is not touched: that is contamination, C4's job.
  C4  C3 plus a description filter supplied by the C4 owner via
      :func:`set_c4_description_filter`.

Query-answer cache: LightRAG's answer cache key does not include who asked, so
a stock cache would hand an exec's answer to a public user. Here the key for
cache_type "query" is salted with the configuration and the user's access set
(C1 answers are shared across users, since C1 context is identical for all).
The keyword-extraction cache depends only on the query text and stays shared,
so all users of a query get the same keywords.
"""
from __future__ import annotations

import contextvars
import dataclasses
import enum
import hashlib
import logging
from contextlib import contextmanager
from typing import Any, Awaitable, Callable, Iterator

from .acl import ACLIndex, split_source_ids, GRAPH_FIELD_SEP

log = logging.getLogger(__name__)


class Mode(str, enum.Enum):
    C1 = "C1"  # no ACL
    C2 = "C2"  # post-retrieval dropout
    C3 = "C3"  # ACL-union pre-filtering
    C4 = "C4"  # C3 + fragment-level description filtering

    @classmethod
    def parse(cls, value: "str | Mode") -> "Mode":
        if isinstance(value, Mode):
            return value
        try:
            return cls(str(value).upper())
        except ValueError:
            raise ValueError(f"unknown config {value!r}; use one of C1, C2, C3, C4") from None

    @property
    def prefilter(self) -> bool:
        return self in (Mode.C3, Mode.C4)


@dataclasses.dataclass(frozen=True)
class Scope:
    mode: Mode
    user_id: str
    access_set: str
    c4_filter: Any = dataclasses.field(default=None, repr=False, compare=False)
    c4_revision: str = ""

    @property
    def cache_scope(self) -> str:
        if self.mode is Mode.C1:
            return self.mode.value
        base = f"{self.mode.value}|{self.access_set}"
        return f"{base}|{self.c4_revision}" if self.mode is Mode.C4 else base


_SCOPE: contextvars.ContextVar[Scope | None] = contextvars.ContextVar("acl_scope", default=None)

# C4 hook: fn(kind, key, data, scope, acl) -> data. kind is "node" or "edge";
# key is the entity name or the (src, tgt) pair. Must return a (possibly
# rewritten) copy of data. Async.
C4Filter = Callable[[str, Any, dict, Scope, ACLIndex], Awaitable[dict]]
_C4_FILTER: C4Filter | None = None


def set_c4_description_filter(fn: C4Filter | None, *, acl: ACLIndex | None = None) -> None:
    """Bind C4 to one ACL index; the legacy global hook remains supported."""
    if acl is not None:
        acl.c4_description_filter = fn
        return
    global _C4_FILTER
    _C4_FILTER = fn


def current_scope() -> Scope | None:
    return _SCOPE.get()


@contextmanager
def acting_as(acl: ACLIndex, user_id: str, mode: "str | Mode") -> Iterator[Scope]:
    """Run the enclosed LightRAG query as ``user_id`` under configuration ``mode``."""
    m = Mode.parse(mode)
    c4_filter = getattr(acl, "c4_description_filter", _C4_FILTER) if m is Mode.C4 else None
    if m is Mode.C4 and c4_filter is None:
        raise NotImplementedError("C4 needs set_c4_description_filter(...) (C4 owner)")
    revision = str(getattr(c4_filter, "cache_namespace", f"custom-{id(c4_filter)}")) if c4_filter else ""
    scope = Scope(m, user_id, acl.access_set_of(user_id), c4_filter, revision)
    token = _SCOPE.set(scope)
    try:
        yield scope
    finally:
        _SCOPE.reset(token)


def _salt(args_hash: str, scope: Scope) -> str:
    return hashlib.md5(f"{args_hash}|acl:{scope.cache_scope}".encode()).hexdigest()


# --------------------------------------------------------------------------- C3
def _install_prefilters(acl: ACLIndex, *, entities_vdb, relationships_vdb, chunks_vdb,
                        graph, text_chunks, max_candidates: int) -> None:

    def active() -> Scope | None:
        s = _SCOPE.get()
        return s if s is not None and s.mode.prefilter else None

    async def readable_chunks(chunk_ids: list[str], user: str) -> list[str]:
        acls = await acl.chunk_acls(chunk_ids)
        return [c for c in chunk_ids if user in acls.get(c, ())]

    async def trim_sources(data: dict, user: str) -> dict:
        ids = split_source_ids(data.get("source_id"))
        if not ids:
            return data
        keep = await readable_chunks(ids, user)
        if len(keep) == len(ids):
            return data
        out = dict(data)
        out["source_id"] = GRAPH_FIELD_SEP.join(keep)
        return out

    async def c4(kind: str, key: Any, data: dict, s: Scope) -> dict:
        if s.mode is Mode.C4 and s.c4_filter is not None:
            return await s.c4_filter(kind, key, data, s, acl)
        return data

    # ---- vector stores: oversample, filter, cut to top_k
    def wrap_query(vdb, is_ok: Callable[[dict, str], Awaitable[bool]]):
        orig = vdb.query

        async def query(query, top_k=None, *args, **kwargs):
            s = active()
            if s is None or not top_k:
                return await orig(query, top_k, *args, **kwargs)
            k = top_k
            while True:
                res = await orig(query, k, *args, **kwargs) or []
                kept = [r for r in res if await is_ok(r, s.user_id)]
                if len(kept) >= top_k or len(res) < k or k >= max_candidates:
                    return kept[:top_k]
                k = min(k * 4, max_candidates)

        vdb.query = query

    async def entity_ok(r: dict, user: str) -> bool:
        return user in await acl.node_acl(r["entity_name"])

    async def relation_ok(r: dict, user: str) -> bool:
        return user in await acl.edge_acl(r["src_id"], r["tgt_id"])

    async def chunk_ok(r: dict, user: str) -> bool:
        cid = r.get("id") or r.get("chunk_id")
        acl.note_chunk_doc(cid, r.get("full_doc_id"))
        return user in (await acl.chunk_acls([cid])).get(cid, ())

    wrap_query(entities_vdb, entity_ok)
    wrap_query(relationships_vdb, relation_ok)
    if chunks_vdb is not None:
        wrap_query(chunks_vdb, chunk_ok)

        orig_vecs = chunks_vdb.get_vectors_by_ids

        async def get_vectors_by_ids(ids, *a, **kw):
            out = await orig_vecs(ids, *a, **kw)
            s = active()
            if s is None or not out:
                return out
            keep = set(await readable_chunks(list(out), s.user_id))
            return {k: v for k, v in out.items() if k in keep}

        chunks_vdb.get_vectors_by_ids = get_vectors_by_ids

    # ---- graph reads used by retrieval and one-hop expansion
    orig_nodes = graph.get_nodes_batch

    async def get_nodes_batch(node_ids, *a, **kw):
        out = await orig_nodes(node_ids, *a, **kw)
        s = active()
        if s is None or not out:
            return out
        res = {}
        for name, data in out.items():
            if data is None or s.user_id not in await acl.node_acl(name, data):
                continue
            res[name] = await c4("node", name, await trim_sources(data, s.user_id), s)
        return res

    orig_node_edges = graph.get_nodes_edges_batch

    async def get_nodes_edges_batch(node_ids, *a, **kw):
        out = await orig_node_edges(node_ids, *a, **kw)
        s = active()
        if s is None or not out:
            return out
        res = {}
        for name, edges in out.items():
            res[name] = [e for e in (edges or []) if s.user_id in await acl.edge_acl(e[0], e[1])]
        return res

    orig_edges = graph.get_edges_batch

    async def get_edges_batch(pairs, *a, **kw):
        out = await orig_edges(pairs, *a, **kw)
        s = active()
        if s is None or not out:
            return out
        res = {}
        for pair, data in out.items():
            if data is None or s.user_id not in await acl.edge_acl(pair[0], pair[1], data):
                continue
            res[pair] = await c4("edge", pair, await trim_sources(data, s.user_id), s)
        return res

    graph.get_nodes_batch = get_nodes_batch
    graph.get_nodes_edges_batch = get_nodes_edges_batch
    graph.get_edges_batch = get_edges_batch

    # ---- chunk text reads
    orig_get_by_ids = text_chunks.get_by_ids

    async def get_by_ids(ids, *a, **kw):
        out = await orig_get_by_ids(ids, *a, **kw)
        s = active()
        if s is None:
            return out
        ok = set(await readable_chunks(list(ids), s.user_id))
        return [row if cid in ok else None for cid, row in zip(ids, out)]

    orig_get_by_id = text_chunks.get_by_id

    async def get_by_id(cid, *a, **kw):
        row = await orig_get_by_id(cid, *a, **kw)
        s = active()
        if s is None or row is None:
            return row
        return row if await readable_chunks([cid], s.user_id) else None

    text_chunks.get_by_ids = get_by_ids
    text_chunks.get_by_id = get_by_id


# --------------------------------------------------------------------------- C2 + cache
def _install_module_hooks(acl: ACLIndex, operate) -> None:
    if getattr(operate, "_acl_hooks_installed", False):
        operate._acl_index = acl
        return
    operate._acl_index = acl

    orig_search = operate._perform_kg_search
    orig_merge = operate._merge_all_chunks
    orig_handle_cache = operate.handle_cache
    orig_save_to_cache = operate.save_to_cache

    def c2() -> Scope | None:
        s = _SCOPE.get()
        return s if s is not None and s.mode is Mode.C2 else None

    async def relation_acl(rel: dict) -> frozenset[str]:
        a = operate._acl_index
        if "src_tgt" in rel:
            s, t = rel["src_tgt"]
        else:
            s, t = rel.get("src_id"), rel.get("tgt_id")
        return await a.edge_acl(s, t, rel)

    async def _perform_kg_search(*args, **kwargs):
        res = await orig_search(*args, **kwargs)
        s = c2()
        if s is None:
            return res
        a = operate._acl_index
        u = s.user_id
        ents = [e for e in res["final_entities"] if u in await a.node_acl(e["entity_name"], e)]
        rels = [r for r in res["final_relations"] if u in await relation_acl(r)]
        vec = res["vector_chunks"]
        ids = [c.get("chunk_id") or c.get("id") for c in vec]
        cacl = await a.chunk_acls(i for i in ids if i)
        kept = [c for c, i in zip(vec, ids) if i and u in cacl.get(i, ())]
        tracking = {k: v for k, v in res["chunk_tracking"].items()
                    if not (v.get("source") == "C" and u not in cacl.get(k, (u,)))}
        log.debug("C2 dropout %s: entities %d->%d relations %d->%d chunks %d->%d", u,
                  len(res["final_entities"]), len(ents), len(res["final_relations"]), len(rels),
                  len(vec), len(kept))
        return {**res, "final_entities": ents, "final_relations": rels,
                "vector_chunks": kept, "chunk_tracking": tracking}

    async def _merge_all_chunks(*args, **kwargs):
        chunks = await orig_merge(*args, **kwargs)
        s = c2()
        if s is None or not chunks:
            return chunks
        ids = [c.get("chunk_id") or c.get("id") for c in chunks]
        cacl = await operate._acl_index.chunk_acls(i for i in ids if i)
        return [c for c, i in zip(chunks, ids) if i and s.user_id in cacl.get(i, ())]

    async def handle_cache(hashing_kv, args_hash, prompt, mode="default", cache_type="unknown"):
        s = _SCOPE.get()
        if s is not None and cache_type == "query":
            args_hash = _salt(args_hash, s)
        return await orig_handle_cache(hashing_kv, args_hash, prompt, mode, cache_type)

    async def save_to_cache(hashing_kv, cache_data):
        s = _SCOPE.get()
        if s is not None and getattr(cache_data, "cache_type", None) == "query":
            cache_data = dataclasses.replace(cache_data, args_hash=_salt(cache_data.args_hash, s))
        return await orig_save_to_cache(hashing_kv, cache_data)

    operate._perform_kg_search = _perform_kg_search
    operate._merge_all_chunks = _merge_all_chunks
    operate.handle_cache = handle_cache
    operate.save_to_cache = save_to_cache
    operate._acl_hooks_installed = True


# --------------------------------------------------------------------------- entry point
def install(rag, acl: ACLIndex, *, operate=None, max_candidates: int = 4096) -> ACLIndex:
    """Bind ``acl`` to ``rag``'s stores and install C2/C3 enforcement + cache keying.

    Call once, after ``await rag.initialize_storages()``. Safe for ingestion:
    nothing changes unless a query runs inside :func:`acting_as`.
    """
    if operate is None:
        from lightrag import operate  # noqa: PLC0415
    if getattr(rag, "_acl_installed", False):
        return rag._acl_installed
    graph = rag.chunk_entity_relation_graph
    acl.bind(rag.text_chunks.get_by_ids, graph.get_node, graph.get_edge)  # unpatched readers
    _install_prefilters(
        acl,
        entities_vdb=rag.entities_vdb,
        relationships_vdb=rag.relationships_vdb,
        chunks_vdb=rag.chunks_vdb,
        graph=graph,
        text_chunks=rag.text_chunks,
        max_candidates=max_candidates,
    )
    _install_module_hooks(acl, operate)
    rag._acl_installed = acl
    return acl
