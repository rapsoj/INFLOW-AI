from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import f1_score, mean_absolute_error, precision_score, recall_score, roc_auc_score


@dataclass
class ModelMetrics:
    calibration: float
<<<<<<< HEAD
    ci95_percent_better_than_std: float
=======
>>>>>>> origin/main
    twcrps: float
    mae: float
    rmse: float
    quantile_loss_95: float
    quantile_loss_99: float
    peak_precision: float
    peak_recall: float
    peak_auc: float
    peak_f1: float
<<<<<<< HEAD
    true_event_rate: float
    pred_event_rate: float
=======
>>>>>>> origin/main

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(q * diff, (q - 1.0) * diff)))


def _empirical_crps(y_true: np.ndarray, samples: np.ndarray) -> np.ndarray:
    # samples shape: (n_samples, n_obs)
    first_term = np.mean(np.abs(samples - y_true[np.newaxis, :]), axis=0)
    pairwise = np.abs(samples[:, np.newaxis, :] - samples[np.newaxis, :, :])
    second_term = 0.5 * np.mean(pairwise, axis=(0, 1))
    return first_term - second_term


def _time_weighted_mean(values: np.ndarray) -> float:
    n = len(values)
    if n == 0:
        return float("nan")
    weights = np.arange(1, n + 1, dtype=np.float64)
    weights /= weights.sum()
    return float(np.sum(values * weights))


def _safe_auc(y_true_events: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true_events)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true_events, y_score))


def compute_metrics(
    y_true_raw: np.ndarray,
    y_pred_raw: np.ndarray,
    y_pred_samples_raw: np.ndarray | None = None,
<<<<<<< HEAD
    dry_season_baseline: float | np.ndarray | None = None,
=======
    dry_season_baseline: float | None = None,
>>>>>>> origin/main
    event_change_threshold: float = 0.05,
) -> ModelMetrics:
    y_true = y_true_raw.astype(np.float64)
    y_pred = y_pred_raw.astype(np.float64)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    if y_pred_samples_raw is None:
        y_pred_samples = np.expand_dims(y_pred, axis=0)
    else:
        y_pred_samples = y_pred_samples_raw.astype(np.float64)

    lower_95 = np.quantile(y_pred_samples, 0.025, axis=0)
    upper_95 = np.quantile(y_pred_samples, 0.975, axis=0)
    coverage_95 = np.mean((y_true >= lower_95) & (y_true <= upper_95))

<<<<<<< HEAD
    # Baseline interval: point prediction +/- 1 predictive std (~68% under normality).
    pred_std = np.std(y_pred_samples, axis=0)
    lower_std = y_pred - pred_std
    upper_std = y_pred + pred_std
    coverage_std = np.mean((y_true >= lower_std) & (y_true <= upper_std))

    target_coverage = 0.95
    err_95 = abs(float(coverage_95) - target_coverage)
    err_std = abs(float(coverage_std) - target_coverage)
    if err_std <= 1e-12:
        ci95_percent_better_than_std = 0.0
    else:
        ci95_percent_better_than_std = float(((err_std - err_95) / err_std) * 100.0)

=======
>>>>>>> origin/main
    crps_series = _empirical_crps(y_true, y_pred_samples)
    twcrps = _time_weighted_mean(crps_series)

    q95_pred = np.quantile(y_pred_samples, 0.95, axis=0)
    q99_pred = np.quantile(y_pred_samples, 0.99, axis=0)
    ql95 = _pinball_loss(y_true, q95_pred, q=0.95)
    ql99 = _pinball_loss(y_true, q99_pred, q=0.99)

    if dry_season_baseline is None:
        dry_season_baseline = float(np.nanmin(y_true))

<<<<<<< HEAD
    baseline_arr = np.asarray(dry_season_baseline, dtype=np.float64)
    if baseline_arr.ndim == 0:
        baseline_arr = np.full_like(y_true, float(baseline_arr), dtype=np.float64)
    if baseline_arr.shape != y_true.shape:
        raise ValueError(
            "dry_season_baseline must be scalar or have the same shape as y_true_raw."
        )

    denom = np.maximum(np.abs(baseline_arr), 1e-6)
    rel_true = (y_true - baseline_arr) / denom
    rel_pred = (y_pred - baseline_arr) / denom

    true_events = rel_true >= event_change_threshold
    pred_events = rel_pred >= event_change_threshold
    event_score = rel_pred
=======
    denom = max(abs(dry_season_baseline), 1e-6)
    true_events = ((y_true - dry_season_baseline) / denom) >= event_change_threshold
    pred_events = ((y_pred - dry_season_baseline) / denom) >= event_change_threshold
>>>>>>> origin/main

    peak_precision = float(precision_score(true_events, pred_events, zero_division=0))
    peak_recall = float(recall_score(true_events, pred_events, zero_division=0))
    peak_f1 = float(f1_score(true_events, pred_events, zero_division=0))
<<<<<<< HEAD
    peak_auc = _safe_auc(true_events.astype(int), event_score)

    true_event_rate = float(np.mean(true_events.astype(np.float64)))
    pred_event_rate = float(np.mean(pred_events.astype(np.float64)))

    return ModelMetrics(
        calibration=float(coverage_95),
        ci95_percent_better_than_std=ci95_percent_better_than_std,
=======
    peak_auc = _safe_auc(true_events.astype(int), y_pred)

    return ModelMetrics(
        calibration=float(coverage_95),
>>>>>>> origin/main
        twcrps=twcrps,
        mae=mae,
        rmse=rmse,
        quantile_loss_95=ql95,
        quantile_loss_99=ql99,
        peak_precision=peak_precision,
        peak_recall=peak_recall,
        peak_auc=peak_auc,
        peak_f1=peak_f1,
<<<<<<< HEAD
        true_event_rate=true_event_rate,
        pred_event_rate=pred_event_rate,
=======
>>>>>>> origin/main
    )
