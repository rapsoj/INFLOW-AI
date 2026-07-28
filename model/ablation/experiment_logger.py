from __future__ import annotations

import os
from typing import Any

import pandas as pd


class ExperimentLogger:
    def __init__(self, log_csv_path: str):
        self.log_csv_path = log_csv_path
        os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)

    def append(self, row: dict[str, Any]) -> None:
        row_df = pd.DataFrame([row])
        if os.path.exists(self.log_csv_path):
            existing = pd.read_csv(self.log_csv_path)
            union_cols = sorted(set(existing.columns).union(set(row_df.columns)))
            existing = existing.reindex(columns=union_cols)
            row_df = row_df.reindex(columns=union_cols)
            combined = pd.concat([existing, row_df], ignore_index=True)
        else:
            combined = row_df

        combined.to_csv(self.log_csv_path, index=False)
