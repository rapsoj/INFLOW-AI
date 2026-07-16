"""Operational wrapper for running the full INFLOW-AI pipeline."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inflow_ai.pipelines.main import main


if __name__ == "__main__":
    main()