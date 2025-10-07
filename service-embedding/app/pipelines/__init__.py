"""Embedding processing pipelines.

The modules in this package orchestrate end‑to‑end embedding workflows.
They define reusable steps for pre‑processing text, invoking the encoder,
post‑processing vectors, and handling transient failures with retry/backoff.

Highlights
- Clear separation of concerns between preprocessing, encoding, and I/O
- Centralized retry and error‑handling helpers for robust pipelines
"""
