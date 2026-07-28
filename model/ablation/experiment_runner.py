from __future__ import annotations

import itertools
import os
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Iterable

import numpy as np

from .data_pipeline import TargetType, prepare_dataset
from .experiment_logger import ExperimentLogger
from .metrics import compute_metrics
from .models import MODEL_REGISTRY
from .utils import set_global_seed

LOG_PATH = "model/ablation/ablation_experiment_log.csv"
WEIGHTS_DIR = "model/ablation/models/weights"


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


def _weights_path(model_type: str, inundation_product: str, target_type: str, autoregressive: bool, cutoff: str, run_id: str) -> str:
    safe_cutoff = cutoff.replace("-", "")
    ar_flag = "ar" if autoregressive else "noar"
    filename = f"{model_type}_{inundation_product}_{target_type}_{ar_flag}_{safe_cutoff}_{run_id}.pkl"
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    return os.path.join(WEIGHTS_DIR, filename)


def _as_log_row(config: AblationConfig, run_id: str, status: str, payload: dict[str, Any]) -> dict[str, Any]:
    base = {
        "run_id": run_id,
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "status": status,
        **asdict(config),
    }
    base.update(payload)
    return base


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

        dataset = prepare_dataset(
            inundation_product=config.inundation_product,
            target_type=config.target_type,
            autoregressive=config.autoregressive,
            training_cutoff_date=config.training_cutoff_date,
        )

        model = MODEL_REGISTRY[config.model_type](seed=config.seed)
        model.fit(dataset.X_train, dataset.y_train)

        y_pred_transformed = model.predict(dataset.X_test)
        y_pred_samples_transformed = model.predict_samples(dataset.X_test)

        y_pred_raw = y_pred_transformed + dataset.raw_reconstruction_component_test
        y_pred_samples_raw = y_pred_samples_transformed + dataset.raw_reconstruction_component_test[np.newaxis, :]

        dry_baseline = float(np.nanmin(dataset.y_train_raw))
        metrics = compute_metrics(
            y_true_raw=dataset.y_test_raw,
            y_pred_raw=y_pred_raw,
            y_pred_samples_raw=y_pred_samples_raw,
            dry_season_baseline=dry_baseline,
            event_change_threshold=0.05,
        )

        model_path = _weights_path(
            model_type=config.model_type,
            inundation_product=config.inundation_product,
            target_type=config.target_type,
            autoregressive=config.autoregressive,
            cutoff=config.training_cutoff_date,
            run_id=run_id,
        )
        model.save_weights(model_path)

        payload = {
            "model_weights_path": model_path,
            **dataset.properties,
            **metrics.to_dict(),
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
    results = []
    for config in configs:
        results.append(run_single_ablation(config, logger=logger))
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
