"""Training package for model development and experiments.

This package organizes reusable training utilities, synthetic data generators,
and example pipelines used for model experimentation.

Highlights:
- Curve forecasting example lives in `training.curve_forecaster`.
- Keep heavy dependencies inside concrete modules to keep import time small.

Typical usage:
- from training.curve_forecaster.data_generator import CurveDataGenerator

Notes:
- Avoid side effects at import time; perform slow work behind `if __name__ == "__main__"`.
"""
