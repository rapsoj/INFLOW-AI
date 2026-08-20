from __future__ import annotations

import os
from typing import Any

import pandas as pd


def _ordered_columns(columns: list[str]) -> list[str]:
    col_set = set(columns)

    core_priority = [
        "run_id",
        "timestamp_utc",
        "status",
        "model_type",
        "training_cutoff_date",
        "inundation_product",
        "target_type",
        "autoregressive",
        "seed",
        "dataset_forecast_horizon_mode",
        "dataset_forecast_horizon_periods",
        "dataset_forecast_horizon_days_approx",
        "dataset_inferred_period_days",
        "dataset_autoregressive_requested",
        "dataset_target_lag_feature_toggle",
        "dataset_target_lag_features_used",
        "dataset_target_lag_steps",
        "dataset_feature_lag_feature_toggle",
        "dataset_feature_lag_features_used",
        "dataset_feature_lag_steps",
        "feature_selection_enabled",
        "feature_selection_best_k",
        "selected_feature_count",
    ]

    metric_priority = [
        "metrics_target_space",
        "mae",
        "rmse",
        "calibration",
        "twcrps",
        "quantile_loss_95",
        "quantile_loss_99",
        "peak_precision",
        "peak_recall",
        "peak_auc",
        "peak_f1",
        "metrics_by_lead_time",
    ]

    last_priority = [
        "model_weights_path",
        "dataset_scaler_path",
        "dataset_pca_path",
        "feature_selection_cache_path",
        "dataset_source_path",
        "dataset_source_files",
        "error_type",
        "error_message",
        "traceback",
    ]

    ordered: list[str] = []

    for col in core_priority:
        if col in col_set and col not in ordered:
            ordered.append(col)
    for col in metric_priority:
        if col in col_set and col not in ordered:
            ordered.append(col)

    remaining = [c for c in columns if c not in ordered]

    # Any error/path/cache/artifact diagnostics should be placed at the end.
    diagnostic_markers = ("path", "error", "traceback", "cache", "source_file", "weights")
    remaining_non_diag = sorted(
        [c for c in remaining if not any(marker in c for marker in diagnostic_markers)]
    )
    remaining_diag = sorted([c for c in remaining if any(marker in c for marker in diagnostic_markers)])

    for col in remaining_non_diag:
        if col not in ordered:
            ordered.append(col)
    for col in last_priority:
        if col in col_set and col not in ordered:
            ordered.append(col)
    for col in remaining_diag:
        if col not in ordered:
            ordered.append(col)

    return ordered


class ExperimentLogger:
    def __init__(self, log_csv_path: str):
        self.log_csv_path = log_csv_path
        os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)

    def append(self, row: dict[str, Any]) -> None:
        row_df = pd.DataFrame([row])
        if os.path.exists(self.log_csv_path):
            existing = pd.read_csv(self.log_csv_path)
            union_cols = _ordered_columns(list(set(existing.columns).union(set(row_df.columns))))
            existing = existing.reindex(columns=union_cols)
            row_df = row_df.reindex(columns=union_cols)
            combined = pd.concat([existing, row_df], ignore_index=True)
        else:
            union_cols = _ordered_columns(list(row_df.columns))
            combined = row_df.reindex(columns=union_cols)

        combined.to_csv(self.log_csv_path, index=False)
