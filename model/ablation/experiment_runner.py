from __future__ import annotations

import itertools
import json
import os
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from processing.config import get_cfg
from .data_pipeline import TargetType, prepare_dataset, reconstruct_raw_from_transformed
from .experiment_logger import ExperimentLogger
from .feature_selection import select_features_with_cv
from .metrics import compute_metrics
from .models import MODEL_REGISTRY
from .utils import set_global_seed

ARTIFACTS_DIR = Path(get_cfg("ablation.artifacts.base_dir", "model/ablation/artifacts"))
LOG_PATH = str(get_cfg("ablation.experiments.log_path", "model/ablation/ablation_experiment_log.csv"))
WEIGHTS_DIR = ARTIFACTS_DIR / "experiment_weights"
BEST_MODEL_INFO_PATH = Path("model") / "best_temporal_model.json"


@dataclass
class AblationConfig:
    model_type: str
    training_cutoff_date: str
    autoregressive: bool
    target_type: TargetType
    inundation_product: str
    seed: int = 42


def _run_id() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _weights_path(
    model_type: str,
    inundation_product: str,
    target_type: str,
    autoregressive: bool,
    cutoff: str,
    run_id: str,
    lead_period: int | None = None,
) -> str:
    safe_cutoff = cutoff.replace("-", "")
    ar_flag = "ar" if autoregressive else "noar"
    lead_tag = f"_lead{int(lead_period)}" if lead_period is not None else ""
    filename = (
        f"{model_type}_{inundation_product}_{target_type}_{ar_flag}_{safe_cutoff}{lead_tag}_{run_id}.pkl"
    )
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    return str(WEIGHTS_DIR / filename)


def _write_best_model_metadata(row: dict[str, Any]) -> None:
    BEST_MODEL_INFO_PATH.parent.mkdir(parents=True, exist_ok=True)
    BEST_MODEL_INFO_PATH.write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")


def _as_log_row(config: AblationConfig, run_id: str, status: str, payload: dict[str, Any]) -> dict[str, Any]:
    base = {
        "run_id": run_id,
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "status": status,
        **asdict(config),
    }
    base.update(payload)
    return base


def _progress_bar(current: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return "[complete]"
    filled = max(0, min(width, int(round(width * current / total))))
    remaining = width - filled
    return f"[{'#' * filled}{'-' * remaining}]"


def format_progress_message(current: int, total: int, config: AblationConfig) -> str:
    bar = _progress_bar(current, total)
    return (
        f"Progress {current}/{total} {bar} | "
        f"model={config.model_type} | cutoff={config.training_cutoff_date} | "
        f"ar={config.autoregressive} | target={config.target_type} | product={config.inundation_product}"
    )


def _rolling_dry_baseline_past_365_days(
    history_dates: np.ndarray,
    history_target_raw: np.ndarray,
    test_dates: np.ndarray,
) -> np.ndarray:
    """Dry baseline per test timestamp = min observed inundation over prior 365 days."""
    hist = pd.DataFrame({
        "date": pd.to_datetime(history_dates),
        "target_raw": history_target_raw.astype(np.float64),
    }).sort_values("date")

    hist = hist.dropna(subset=["date", "target_raw"]).drop_duplicates(subset=["date"], keep="last")
    hist = hist.set_index("date")

    # Use only historical values strictly before each timestamp.
    rolling_min = hist["target_raw"].rolling("365D", closed="left").min()
    fallback_min = hist["target_raw"].expanding(min_periods=1).min().shift(1)
    baseline = rolling_min.fillna(fallback_min)

    aligned = baseline.reindex(pd.to_datetime(test_dates))
    # Last safety fallback for any unresolved gaps.
    aligned = aligned.fillna(float(hist["target_raw"].min()))
    return aligned.to_numpy(dtype=np.float64)


def _resolve_extreme_event_top_fraction() -> float:
    top_fraction = float(get_cfg("ablation.pipeline.metrics.extreme_event_top_fraction", 0.10))
    if top_fraction <= 0.0 or top_fraction >= 1.0:
        raise ValueError("ablation.pipeline.metrics.extreme_event_top_fraction must be in (0, 1).")
    return top_fraction


def _derive_event_threshold_from_training_rel_true(
    y_train_raw: np.ndarray,
    train_baseline: np.ndarray,
    top_fraction: float,
) -> float:
    denom = np.maximum(np.abs(train_baseline.astype(np.float64)), 1e-6)
    rel_true_train = (y_train_raw.astype(np.float64) - train_baseline.astype(np.float64)) / denom
    valid_rel = rel_true_train[np.isfinite(rel_true_train)]
    if valid_rel.size == 0:
        raise RuntimeError("Unable to derive event threshold: no finite rel_true values in training set.")

    quantile = 1.0 - float(top_fraction)
    return float(np.quantile(valid_rel, quantile))


def _average_metrics_across_leads(lead_metrics: dict[int, dict[str, float]]) -> dict[str, float]:
    if not lead_metrics:
        return {}

    metric_names = sorted(next(iter(lead_metrics.values())).keys())
    averaged: dict[str, float] = {}
    for metric_name in metric_names:
        values = np.array(
            [lead_metrics[lead][metric_name] for lead in sorted(lead_metrics.keys())],
            dtype=np.float64,
        )
        finite_values = values[np.isfinite(values)]
        averaged[metric_name] = float(np.mean(finite_values)) if finite_values.size > 0 else float("nan")
    return averaged


def _flatten_metrics_by_lead(lead_metrics: dict[int, dict[str, float]]) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for lead in sorted(lead_metrics.keys()):
        for metric_name, value in lead_metrics[lead].items():
            flattened[f"lead_{lead}_{metric_name}"] = float(value)
    return flattened


def run_single_ablation(config: AblationConfig, logger: ExperimentLogger | None = None) -> dict[str, Any]:
    if logger is None:
        logger = ExperimentLogger(LOG_PATH)

    run_id = _run_id()
    set_global_seed(config.seed)

    try:
        if config.model_type not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model_type '{config.model_type}'. "
                f"Available: {sorted(MODEL_REGISTRY.keys())}"
            )

        configured_dataset = prepare_dataset(
            inundation_product=config.inundation_product,
            target_type=config.target_type,
            autoregressive=config.autoregressive,
            training_cutoff_date=config.training_cutoff_date,
            run_id=run_id,
            model_type=config.model_type,
        )
        max_lead_period = int(configured_dataset.properties.get("dataset_forecast_horizon_periods", 1))
        if max_lead_period < 1:
            raise RuntimeError("Configured forecast horizon periods resolved to < 1.")

        event_top_fraction = _resolve_extreme_event_top_fraction()
        lead_metrics_by_period: dict[int, dict[str, float]] = {}
        lead_metrics_payload: dict[str, dict[str, float]] = {}
        model_weights_by_lead: dict[str, str] = {}
        extreme_threshold_by_lead: dict[str, float] = {}
        feature_selection_best_k_by_lead: dict[str, int] = {}
        selected_feature_count_by_lead: dict[str, int] = {}
        selected_feature_names_by_lead: dict[str, str] = {}

        feature_selection_enabled_all = True
        feature_selection_used_cache_all = True
        reference_feature_selection = None

        for lead_period in range(1, max_lead_period + 1):
            dataset = prepare_dataset(
                inundation_product=config.inundation_product,
                target_type=config.target_type,
                autoregressive=config.autoregressive,
                training_cutoff_date=config.training_cutoff_date,
                run_id=run_id,
                model_type=config.model_type,
                forecast_horizon_periods_override=lead_period,
            )

            feature_selection = select_features_with_cv(
                X_train=dataset.X_train,
                X_test=dataset.X_test,
                y_train=dataset.y_train,
                feature_names=dataset.feature_names,
                model_type=config.model_type,
                seed=config.seed,
                dataset_fingerprint=str(dataset.properties.get("dataset_fingerprint", "unknown")),
                configuration_signature={
                    "model_type": config.model_type,
                    "training_cutoff_date": config.training_cutoff_date,
                    "autoregressive": config.autoregressive,
                    "target_type": config.target_type,
                    "inundation_product": config.inundation_product,
                    "lead_period": lead_period,
                },
            )

            if reference_feature_selection is None:
                reference_feature_selection = feature_selection

            feature_selection_enabled_all = feature_selection_enabled_all and feature_selection.enabled
            feature_selection_used_cache_all = feature_selection_used_cache_all and feature_selection.used_cache

            model = MODEL_REGISTRY[config.model_type](seed=config.seed)
            model.fit(feature_selection.X_train, dataset.y_train)

            y_pred_transformed = model.predict(feature_selection.X_test)
            y_pred_samples_transformed = model.predict_samples(feature_selection.X_test)

            y_pred_raw = reconstruct_raw_from_transformed(
                y_pred_transformed=y_pred_transformed,
                raw_reconstruction_component=dataset.raw_reconstruction_component_test,
            )
            y_pred_samples_raw = reconstruct_raw_from_transformed(
                y_pred_transformed=y_pred_samples_transformed,
                raw_reconstruction_component=dataset.raw_reconstruction_component_test,
            )

            dry_baseline = _rolling_dry_baseline_past_365_days(
                history_dates=dataset.history_dates,
                history_target_raw=dataset.history_target_raw,
                test_dates=dataset.test_dates,
            )
            train_baseline = _rolling_dry_baseline_past_365_days(
                history_dates=dataset.history_dates,
                history_target_raw=dataset.history_target_raw,
                test_dates=dataset.train_dates,
            )
            event_change_threshold = _derive_event_threshold_from_training_rel_true(
                y_train_raw=dataset.y_train_raw,
                train_baseline=train_baseline,
                top_fraction=event_top_fraction,
            )

            metrics = compute_metrics(
                y_true_raw=dataset.y_test_raw,
                y_pred_raw=y_pred_raw,
                y_pred_samples_raw=y_pred_samples_raw,
                dry_season_baseline=dry_baseline,
                event_change_threshold=event_change_threshold,
            )
            lead_metrics_by_period[lead_period] = metrics.to_dict()
            lead_metrics_payload[f"lead_periods_{lead_period}"] = metrics.to_dict()

            model_path = _weights_path(
                model_type=config.model_type,
                inundation_product=config.inundation_product,
                target_type=config.target_type,
                autoregressive=config.autoregressive,
                cutoff=config.training_cutoff_date,
                run_id=run_id,
                lead_period=lead_period,
            )
            model.save_weights(model_path)

            model_weights_by_lead[str(lead_period)] = model_path
            extreme_threshold_by_lead[str(lead_period)] = float(event_change_threshold)
            feature_selection_best_k_by_lead[str(lead_period)] = int(feature_selection.best_k)
            selected_feature_count_by_lead[str(lead_period)] = int(len(feature_selection.selected_feature_names))
            selected_feature_names_by_lead[str(lead_period)] = "|".join(feature_selection.selected_feature_names)

        averaged_metrics = _average_metrics_across_leads(lead_metrics_by_period)
        flattened_lead_metrics = _flatten_metrics_by_lead(lead_metrics_by_period)

        if reference_feature_selection is None:
            raise RuntimeError("Internal error: no lead-specific feature selection results were computed.")

        payload = {
            "model_weights_path": model_weights_by_lead.get(str(max_lead_period), ""),
            "model_weights_path_by_lead": json.dumps(model_weights_by_lead, sort_keys=True),
            "dry_baseline_definition": "rolling_min_past_365_days",
            "extreme_event_definition": "rel_true_above_training_quantile_threshold",
            "extreme_event_top_fraction": event_top_fraction,
            "extreme_event_threshold": extreme_threshold_by_lead.get(str(max_lead_period), float("nan")),
            "extreme_event_threshold_by_lead": json.dumps(extreme_threshold_by_lead, sort_keys=True),
            "metrics_target_space": "raw",
            "metrics_averaging_scope": "equal_weight_across_lead_periods",
            "metrics_by_lead_time": json.dumps(lead_metrics_payload, sort_keys=True),
            "feature_selection_enabled": feature_selection_enabled_all,
            "feature_selection_used_cache": feature_selection_used_cache_all,
            "feature_selection_cache_path": reference_feature_selection.cache_path,
            "feature_selection_best_k": feature_selection_best_k_by_lead.get(str(max_lead_period), -1),
            "feature_selection_best_k_by_lead": json.dumps(feature_selection_best_k_by_lead, sort_keys=True),
            "selected_feature_count": selected_feature_count_by_lead.get(str(max_lead_period), 0),
            "selected_feature_count_by_lead": json.dumps(selected_feature_count_by_lead, sort_keys=True),
            "selected_feature_names": selected_feature_names_by_lead.get(str(max_lead_period), ""),
            "selected_feature_names_by_lead": json.dumps(selected_feature_names_by_lead, sort_keys=True),
            "removed_constant_features": "|".join(reference_feature_selection.removed_constant_features),
            "removed_correlated_features": "|".join(reference_feature_selection.removed_correlated_features),
            "feature_selection_cv_mean_mae_by_k": json.dumps(
                {str(k): float(v) for k, v in reference_feature_selection.cv_mean_mae_by_k.items()},
                sort_keys=True,
            ),
            "dataset_multi_lead_enabled": True,
            "dataset_lead_period_min": 1,
            "dataset_lead_period_max": int(max_lead_period),
            "dataset_forecast_horizon_periods": int(max_lead_period),
            **configured_dataset.properties,
            **averaged_metrics,
            **{f"avg_{k}": v for k, v in averaged_metrics.items()},
            **flattened_lead_metrics,
        }
        row = _as_log_row(config, run_id, "success", payload)
        logger.append(row)
        return row

    except Exception as e:
        error_payload = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(limit=5),
        }
        row = _as_log_row(config, run_id, "failed", error_payload)
        logger.append(row)
        return row


def run_ablation_grid(configs: Iterable[AblationConfig], log_csv_path: str = LOG_PATH) -> list[dict[str, Any]]:
    logger = ExperimentLogger(log_csv_path)
    configs = list(configs)
    results = []

    total = len(configs)
    for index, config in enumerate(configs, start=1):
        status_label = "running"
        progress_message = format_progress_message(index, total, config)
        print(f"\r{progress_message} | status={status_label}", end="", flush=True)
        result = run_single_ablation(config, logger=logger)
        status_label = result.get("status", "unknown")
        print(f"\r{progress_message} | status={status_label}", end="", flush=True)
        results.append(result)

    print()
    successful = [row for row in results if row.get("status") == "success"]
    if successful:
        best = max(
            successful,
            key=lambda row: (
                float(row.get("avg_peak_f1", row.get("peak_f1", float("-inf")))),
                float(row.get("avg_peak_auc", row.get("peak_auc", float("-inf")))),
            ),
        )
        _write_best_model_metadata(best)
    return results


def build_grid(
    model_types: list[str],
    cutoff_dates: list[str],
    autoregressive_values: list[bool],
    target_types: list[TargetType],
    inundation_products: list[str],
    seed: int,
) -> list[AblationConfig]:
    configs = []
    for model_type, cutoff, autoregressive, target_type, product in itertools.product(
        model_types,
        cutoff_dates,
        autoregressive_values,
        target_types,
        inundation_products,
    ):
        configs.append(
            AblationConfig(
                model_type=model_type,
                training_cutoff_date=cutoff,
                autoregressive=autoregressive,
                target_type=target_type,
                inundation_product=product,
                seed=seed,
            )
        )
    return configs
