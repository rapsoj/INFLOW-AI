from __future__ import annotations

import json
from typing import Any


def _lead_metrics(row: dict[str, Any]) -> list[dict[str, Any]]:
    raw_payload = row.get("metrics_by_lead_time", "") or ""
    if not raw_payload:
        return []
    try:
        payload = json.loads(raw_payload)
    except (TypeError, json.JSONDecodeError):
        return []

    lead_metrics: list[dict[str, Any]] = []
    for key in sorted(payload.keys()):
        lead_payload = payload[key]
        if isinstance(lead_payload, dict):
            lead_metrics.append({"lead": key, **lead_payload})
    return lead_metrics


def _meets_criteria(
    row: dict[str, Any],
    runtime_target_product: str,
    runtime_target_column: str,
    min_calibration: float,
    min_roc_auc: float,
) -> bool:
    if row.get("status") != "success":
        return False
    if str(row.get("inundation_product", "")).lower() != str(runtime_target_product).lower():
        return False
    if str(row.get("dataset_target_column", "")).strip() != str(runtime_target_column).strip():
        return False

    lead_metrics = _lead_metrics(row)
    if not lead_metrics:
        return False

    for metrics in lead_metrics:
        calibration = float(metrics.get("calibration", float("nan")))
        roc_auc = float(metrics.get("peak_auc", float("nan")))
        if not (calibration >= min_calibration and roc_auc >= min_roc_auc):
            return False

    return True


def select_best_model(
    rows: list[dict[str, Any]],
    runtime_target_product: str,
    runtime_target_column: str,
    min_calibration: float = 0.75,
    min_roc_auc: float = 0.80,
) -> dict[str, Any] | None:
    eligible = [
        row
        for row in rows
        if _meets_criteria(row, runtime_target_product, runtime_target_column, min_calibration, min_roc_auc)
    ]
    if not eligible:
        return None

    def sort_key(row: dict[str, Any]) -> tuple[float, float, float, str]:
        lead_metrics = _lead_metrics(row)
        if not lead_metrics:
            return (-float("inf"), -float("inf"), -float("inf"), "")
        last_metrics = lead_metrics[-1]
        return (
            -float(last_metrics.get("peak_f1", float("nan"))),
            -float(row.get("peak_f1", float("nan"))),
            -float(last_metrics.get("peak_auc", float("nan"))),
            str(row.get("model_type", "")),
        )

    return sorted(eligible, key=sort_key)[0]
