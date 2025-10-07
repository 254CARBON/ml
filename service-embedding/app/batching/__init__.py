"""Batching components for the embedding service.

This package contains helpers used to run embedding inference efficiently by
controlling how inputs are grouped and scheduled onto available compute
resources (CPU, CUDA GPUs, or Apple MPS).

Key pieces
- ``gpu_detector``: Detects available accelerators and recommends devices
  and batch sizes at runtime.

Typical usage
- Import batching utilities from this package in the service startup to pick
  the optimal device and tune batch sizes for the selected model.
"""
