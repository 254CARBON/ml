"""Search retrievers for lexical, semantic, and hybrid workflows.

Retrievers encapsulate how candidates are fetched from backends (e.g.,
OpenSearch, PgVector) before ranking. Splitting retrieval from ranking keeps
pipelines modular and testable.
"""
