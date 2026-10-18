# AMS691
Permission-scoped retrieval for graph RAG: a LightRAG fork with ACL propagation, enforcement configurations C1–C4, and a synthetic multi-tenant corpus with canary facts for measuring leakage and contamination. AMS 691 group project.

- [Corpus and ground truth (Arnav)](DESIGN_corpus.md)
- [C1–C3, cache keys, runner, and metrics (Bala)](DESIGN_enforcement.md)
- [C4 source-fragment filtering, setup, and validation (Sai)](DESIGN_c4.md)

Install `requirements-dev.txt` in a Python virtual environment and run
`python -m pytest -q tests` for offline checks. C4's default ingestion captures
original descriptions before merging; older indices need fresh ingestion.
The C4 design notes include commands for the shared runner and explain which
live experiments remain to be run with the team's backend.
