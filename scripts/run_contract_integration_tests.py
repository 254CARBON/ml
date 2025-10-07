#!/usr/bin/env python3
"""Run contract and integration tests with deterministic pre-flight setup."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import List

from scripts.prepare_test_env import main as prepare_environment


def _run_pytest(pytest_args: List[str]) -> int:
    """Execute pytest with the provided arguments inside the container."""
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TZ", "UTC")

    process = subprocess.run(["pytest", *pytest_args], env=env, check=False)
    return process.returncode


def main() -> int:
    """Prepare dependencies, then run pytest."""
    prepare_environment()
    pytest_args = [
        "tests/contract",
        "tests/integration",
        "-m",
        "contract or integration",
        "-v",
        "--tb=short",
    ]
    return _run_pytest(pytest_args)


if __name__ == "__main__":
    sys.exit(main())
