# C4: descriptions from authorized source fragments

Owner: Sai Sony Malapati. Implementation and offline integration checks complete
on October 18, 2026. Full dev-corpus experiments with the team's actual LLM and
embedding backend are still pending.

This change builds on Bala Ramya's C1/C2/C3, access-set cache keys, runner, and
metrics at commit `3a70f8c7abd98f1cf06c5413387edb7f8b3611b0`. It adds C4 to that
runner rather than implementing another experiment runner. Corpus generation
and structural-leakage/contamination scoring remain Arnav's components.

## Behavior

A shared entity may have a public source and a restricted source. The union
ACL makes that entity visible to a public user, but C3 still returns its merged
description. C4 rebuilds the description from only the user's readable sources.

| Item | C3 | C4 |
|---|---|---|
| Entity or relation visibility | Union of source-chunk ACLs | Same |
| Source IDs | Readable chunks only | Readable chunks represented in the permitted fragments |
| Description | Globally merged text | Only authorized original fragments, optionally summarized |
| Relation keywords | Globally merged keywords | Keywords from authorized fragments |
| Description file paths and entity types | Existing graph fields | Rebuilt from authorized fragments |
| Shared graph and vector embeddings | Original index | Same original index |

Example: public text says "Project Atlas is active" and a restricted source says
"its budget code is ZZ-1234-X". A public C4 description contains the public fact;
an executive's C4 description can contain both. C4 never searches for or redacts
canary strings. It uses source permissions, so it also filters ordinary facts.

## Original-fragment capture

`acl_rag/fragments.py` wraps each LightRAG instance's
`_process_extract_entities` method. After extraction returns, but before the
caller merges its results into the shared graph, it saves:

- Entity/relation key and original description; relation keywords and entity type.
- Exactly one source chunk, its `full_doc_id`, and the document's `acl_users`.
- A stable fragment ID and the source content hash.

Records are stored in `c4_fragments.sqlite3` alongside the index. Each processed
chunk's replacement is transactional. Identical descriptions from different
chunks retain separate provenance. Relation-only endpoints get fragments from
their source relation when no separate entity record exists in that chunk.
This uses LightRAG's undirected edge convention.

The capture hook is pinned to `lightrag-hku==1.5.7`. It does not change extraction
results, global descriptions, or embeddings. It supports this project's normal
document `ainsert` path; custom knowledge-graph insertions are outside its scope.

**An old index needs a fresh ingestion with capture enabled.** Splitting a
merged description on `<SEP>` and matching it to the source ID order is unsafe:
the orders differ, and an LLM summary has already blended its sources. Running
`ainsert` again on documents marked processed does not recreate their fragments;
choose a fresh `working_dir` and ingest the corpus there.

## Query integration and caches

`build_acl_rag(cfg, mode="C4")` opens the fragment store read-only and registers
an async `C4DescriptionFilter` with Bala's existing hook, scoped to that ACL index.
It runs for every entity/relation that C3 permits, including one-hop expansion.

The filter requires a captured chunk, matching document provenance, and both
the recorded and current document ACL to authorize the user. Missing provenance,
an empty store, or a different manifest produces an error. There is no fallback
to the globally merged description. Returned dictionaries are copies; the graph
remains unchanged.

`c4.description_strategy` selects:

- `auto` (default): run LightRAG 1.5.7's own description merge/summary policy on
  permitted originals only. The model never receives restricted fragments for
  that user's C4 summary. A single description is returned directly; multiple
  descriptions may be joined or summarized under the configured limits.
- `join`: concatenate permitted descriptions with `<SEP>`. This avoids C4
  summary-model calls, but live document extraction and answer generation still
  require the selected backend. Use a fresh results directory for this variant.

Safe summaries are cached by access set, item, permitted fragment IDs, manifest,
oracle, store fingerprint, policy, strategy, and summary model configuration.
C4's answer-cache scope also includes the C4 revision, so changes to fragment
content or description policy do not reuse older C4 answers. The existing
query-only keyword cache stays shared across users.

`--no-llm-cache` bypasses query-answer caching and both C4 summary caches
(in-memory and persisted LightRAG summaries). Keyword extraction may remain
cached. Report this distinction when measuring latency; this flag does not make
the entire process or vector/OS caches cold. Use concurrency 1 for timing runs.

The runner records `run_id` and `c4_revision` in each result row. It rejects
resuming a file from another configuration, corpus, backend, query settings,
cache policy, or C4 revision. Existing files without `run_id` also require a new
`results_dir`. It does not overwrite or silently combine them. On a query error,
pending queries are cancelled before closing the index and output file.

## Setup and offline verification

Use Python 3.11 or 3.12. Checks below were run with Python 3.12.14 and LightRAG
1.5.7. Run commands from the repository root. No API key is needed for tests.

Windows PowerShell, after cloning/checking out this branch:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q tests
```

macOS/Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q tests
```

For later commands, `python` means the interpreter in that environment. In
PowerShell, use `.\.venv\Scripts\python.exe` directly if the environment is not
activated; in a Unix shell, use `.venv/bin/python`.

## Live experiment commands

1. Configure the team's backend in `configs/rag.yaml`. For OpenAI, explicitly
   install `python -m pip install "openai>=2,<4"` and set `OPENAI_API_KEY` privately
   in the environment. For Ollama, install its client with
   `python -m pip install ollama` and set the local models, dimensions, and host
   in the YAML. Never commit credentials.
2. Keep `c4.capture_fragments: true`. Use a fresh working directory if any prior
   ingestion lacked capture, for example `rag_storage/{corpus_name}_c4_v1`.
   Use one captured index for all four configurations and a fresh results
   directory, for example `results/{corpus_name}_c4_v1_auto`.
3. Ingest once, inspect `failed_docs`, `chunks_with_wrong_parent`, entity coverage,
   and `c4_provenance` in `ingest_report.json`. A successful test fixture does not
   guarantee actual model extraction found every canonical entity or canary.

```bash
python -m acl_rag.ingest --rag-config configs/rag.yaml
python -m acl_rag.run --rag-config configs/rag.yaml --config C4 --limit 16
python -m acl_rag.run --rag-config configs/rag.yaml --config C4
python -m acl_rag.run --rag-config configs/rag.yaml --config C1
python -m acl_rag.run --rag-config configs/rag.yaml --config C2
python -m acl_rag.run --rag-config configs/rag.yaml --config C3
python -m acl_rag.metrics --rag-config configs/rag.yaml
```

The 16-row smoke run resumes into the full run if settings stay identical.
For latency experiments, choose another results directory before running each
configuration with `--concurrency 1 --no-llm-cache`; changing that flag makes an
existing run incompatible. Keep the same flag and concurrency for C1–C4.

Outputs remain Bala's `<results_dir>/C1.jsonl` through `C4.jsonl` and
`C*_metrics.json`. Each C4 row includes served entity/relation descriptions,
source IDs, chunks, answer, latency, and the revision. These are the inputs to
Arnav's separate structural-leakage/contamination analysis. Type-b neighbor
canaries can include contamination canaries; retain that distinction.

## Validation and remaining experimental work

`python -m pytest -q tests` covers the original C1–C3 checks plus C4:

| Check | Evidence and scope |
|---|---|
| All 224 dev query-user rows | Real `kg_query`, in-memory stores, deterministic model, per-document original-fragment fixture; no unauthorized retrieval or forbidden canaries in served C4 context/answers; positive controls retain authorized target canaries |
| Actual ingestion and persistence | Real LightRAG extraction parser, chunk/graph/vector stores, merges, sidecar, factory reopen, and `aquery_llm` on three documents with stub model/embeddings |
| Summary behavior | Real LightRAG summarizer policy, forced via test token limits; public summary inputs contain only public fragments; executive retrieval retains restricted facts |
| Cache isolation | Public/executive query sequence, per-index callbacks, C4 revisions, and explicit summary-cache bypass |
| Error and provenance cases | Missing/ambiguous sources, mismatched manifests, atomic replacement, duplicate text across permissions, incompatible result resumption, cancellation before storage closure |

These are implementation regression checks, **not final empirical results**.
The 224-row fixture uses ground-truth fragments and explicit retrieval keywords;
the real-storage fixture uses only three documents and local model stubs.
Real model extraction can miss facts, change entity names, or omit canaries;
generated summaries/answers can also omit facts or hallucinate. Run the full
50-document dev experiment with the agreed backend before claiming empirical
leakage, contamination, answer-quality, or latency numbers.

Operate on a fixed index/manifest during an experiment. Rebuild after corpus,
permission, deletion, or extraction-policy changes; do not ingest concurrently
with queries. Dynamic revocation/deletion and cache invalidation across mutable
indices are outside this implementation. C4 also retains C3's globally learned
embeddings and graph-degree ranking, so it does not eliminate ranking side
channels or promise perfect recall. Source-count hub flags in the ingestion
report are proxies, not logs proving a particular summary-model call occurred.

Next team steps: run the agreed backend once for all configurations; inspect
canonical-name and canary extraction coverage; compute structural leakage and
contamination separately; then fill the project's experimental results and
presentation from those measured outputs.
