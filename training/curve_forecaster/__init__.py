"""Curve forecasting components and helpers.

Contains:
- ``data_generator.py``: utilities to synthesize realistic yield, commodity,
  and FX curve datasets for downstream forecasting models.

Guidance:
- Keep synthetic data generation and feature construction deterministic by
  using explicit seeds to enable reproducible experiments.
"""
