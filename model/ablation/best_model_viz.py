from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from processing.config import get_cfg
from .data_pipeline import prepare_dataset, reconstruct_raw_from_transformed
from .feature_selection import select_features_with_cv
from .models import MODEL_REGISTRY
from .utils import set_global_seed


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _to_svg_points(values: list[float], width: int, height: int, padding: int = 40) -> str:
    if not values:
        return ""
    min_value = min(values)
    max_value = max(values)
    if min_value == max_value:
        min_value -= 0.5
        max_value += 0.5
    span = max_value - min_value
    points: list[str] = []
    for index, value in enumerate(values):
        x = padding + (index / max(len(values) - 1, 1)) * (width - 2 * padding)
        y = height - padding - ((value - min_value) / span) * (height - 2 * padding)
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def _format_title(row: dict[str, Any], lead_period: int, metrics: dict[str, Any]) -> str:
    return (
        f"Best Model: {row.get('model_type', 'unknown')} | "
        f"run_id={row.get('run_id', 'unknown')} | target_type={row.get('target_type', 'unknown')} | "
        f"autoregressive={row.get('autoregressive', 'unknown')} | "
        f"Lead {lead_period} | F1 {metrics.get('peak_f1', float('nan')):.3f}"
    )


def _compute_predicted_series(row: dict[str, Any], lead_period: int) -> tuple[list[datetime], list[float], list[datetime], list[float], list[datetime], list[float]]:
    training_cutoff_date = str(row.get("training_cutoff_date") or get_cfg("ablation.experiments.training_cutoff_dates", [""])[0])
    seed = int(row.get("seed", get_cfg("ablation.experiments.seed", 42)))
    config_row = {
        "model_type": str(row.get("model_type", "")),
        "target_type": str(row.get("target_type", "")),
        "autoregressive": _coerce_bool(row.get("autoregressive", False)),
        "inundation_product": str(row.get("inundation_product", "viirs")),
        "training_cutoff_date": training_cutoff_date,
        "seed": seed,
    }

    set_global_seed(seed)
    dataset = prepare_dataset(
        inundation_product=config_row["inundation_product"],
        target_type=config_row["target_type"],
        autoregressive=config_row["autoregressive"],
        training_cutoff_date=config_row["training_cutoff_date"],
        run_id=str(row.get("run_id", "")),
        model_type=config_row["model_type"],
        forecast_horizon_periods_override=lead_period,
    )

    feature_selection = select_features_with_cv(
        X_train=dataset.X_train,
        X_test=dataset.X_test,
        y_train=dataset.y_train,
        feature_names=dataset.feature_names,
        model_type=config_row["model_type"],
        seed=seed,
        dataset_fingerprint=str(dataset.properties.get("dataset_fingerprint", "unknown")),
        configuration_signature={
            "model_type": config_row["model_type"],
            "training_cutoff_date": config_row["training_cutoff_date"],
            "autoregressive": config_row["autoregressive"],
            "target_type": config_row["target_type"],
            "inundation_product": config_row["inundation_product"],
            "lead_period": lead_period,
        },
    )

    model = MODEL_REGISTRY[config_row["model_type"]](seed=seed)
    model.fit(feature_selection.X_train, dataset.y_train)
    y_pred_transformed = model.predict(feature_selection.X_test)
    y_pred_raw = reconstruct_raw_from_transformed(
        y_pred_transformed=y_pred_transformed,
        raw_reconstruction_component=dataset.raw_reconstruction_component_test,
    )

    split_date = datetime.strptime(training_cutoff_date, "%Y-%m-%d")
    history_dates = [datetime.fromtimestamp(ts / 1_000_000_000.0) if isinstance(ts, (int, np.integer)) else datetime.fromisoformat(str(ts)) for ts in dataset.history_dates]
    history_values = [float(value) for value in dataset.history_target_raw]

    history_before_split = [
        (date_value, value) for date_value, value in zip(history_dates, history_values) if date_value <= split_date
    ]

    actual_dates = [date_value for date_value, _ in history_before_split]
    actual_values = [value for _, value in history_before_split]
    test_dates = [datetime.fromisoformat(str(ts)) for ts in dataset.test_dates]
    test_actual_values = [float(value) for value in dataset.y_test_raw]
    actual_dates.extend(test_dates)
    actual_values.extend(test_actual_values)

    return actual_dates, actual_values, test_dates, [float(value) for value in y_pred_raw], test_dates, test_actual_values


def _render_svg(output_path: Path, title: str, actual_dates: list[datetime], actual_values: list[float], test_dates: list[datetime], pred_values: list[float], split_date: datetime | None) -> None:
    if not actual_dates:
        raise ValueError("No actual dates available for plotting")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import dates as mdates

    if test_dates:
        first_test_date = test_dates[0]
        prediction_dates: list[datetime] = []
        for current_date in test_dates:
            if split_date is None:
                prediction_dates.append(current_date)
            else:
                prediction_dates.append(split_date + (current_date - first_test_date))
    else:
        prediction_dates = []

    if split_date is None:
        plot_split_date = None
    else:
        plot_split_date = split_date

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(actual_dates, actual_values, color="#1f77b4", linewidth=2.5, label="Actual")
    if prediction_dates and pred_values:
        ax.plot(prediction_dates, pred_values, color="#d62728", linestyle="--", linewidth=2.5, label="Predicted")
    if plot_split_date is not None:
        ax.axvline(plot_split_date, color="#222", linestyle=":", linewidth=1.5, label="Train/Test Split")

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Inundation %")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    if actual_dates and prediction_dates:
        all_dates = actual_dates + prediction_dates
        min_date = min(all_dates)
        max_date = max(all_dates)
        ax.set_xlim(min_date - timedelta(days=30), max_date + timedelta(days=30))

    fig.autofmt_xdate()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    min_value = min(actual_values + pred_values) if pred_values else min(actual_values)
    max_value = max(actual_values + pred_values) if pred_values else max(actual_values)
    if min_value == max_value:
        min_value -= 0.5
        max_value += 0.5
    ax.set_ylim(min_value, max_value)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_best_model_results(row: dict[str, Any], split_date: str | None = None) -> None:
    if not row:
        raise ValueError("A row must be provided to plot")

    payload = row.get("metrics_by_lead_time", "") or ""
    if not payload:
        raise ValueError("The selected row does not contain metrics_by_lead_time")

    try:
        lead_metrics_payload = json.loads(payload)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("metrics_by_lead_time is not valid JSON") from exc

    lead_periods = sorted(int(key.replace("lead_periods_", "")) for key in lead_metrics_payload.keys())
    if not lead_periods:
        raise ValueError("No lead periods were found in metrics_by_lead_time")

    split_dt = datetime.strptime(split_date, "%Y-%m-%d") if split_date else None
    if split_dt is None:
        split_dt = datetime.strptime(str(row.get("training_cutoff_date", "2018-07-21")), "%Y-%m-%d")

    for lead_period in lead_periods:
        metrics = lead_metrics_payload[f"lead_periods_{lead_period}"]
        actual_dates, actual_values, test_dates, pred_values, _, _ = _compute_predicted_series(row, lead_period)
        title = _format_title(row, lead_period, metrics)
        output_path = Path(f"model/ablation/artifacts/best_model_plot_lead{lead_period}.png")
        _render_svg(output_path, title, actual_dates, actual_values, test_dates, pred_values, split_dt)
        print(f"Plot written to {output_path}")
