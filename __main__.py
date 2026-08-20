from __future__ import annotations

import logging
import json
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import pandas as pd

from data.stats import gridded_stats
from model.ablation.area_reconversion import convert_target_fraction_to_km2, target_area_conversion_metadata
from model.ablation.deployment import (
    generate_temporal_forecast,
    load_best_model_row,
    retrain_best_model_on_available_data,
    update_temporal_sources_for_best_model,
    write_best_model_metadata,
)
from explanations.shap_explanations import export_lead_waterfall_plots


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

FORECAST_PERFORMANCE_CSV = Path("predictions") / "forecast_performance.csv"


def _target_display_name(target_column: str) -> str:
    if target_column == "percent_inundation":
        return "Inundated Area"
    if target_column.startswith("percent_inundation_"):
        region_code = target_column.split("percent_inundation_", 1)[1]
        reverse = {v: k for k, v in gridded_stats.region_to_code_dict.items()}
        region_name = reverse.get(region_code, region_code.upper())
        return f"{region_name} Inundated Area"
    return target_column


def _prediction_output_dir(target_column: str, future_dates: list[pd.Timestamp]) -> Path:
    start = future_dates[0].strftime("%Y-%m-%d")
    end = future_dates[-1].strftime("%Y-%m-%d")
    return Path("predictions") / f"temporal_predictions_{target_column}_{start}_to_{end}"


def export_model_report(bundle) -> Path:
    """Write model identity, held-out metrics, and runtime retraining metadata."""
    output_dir = _prediction_output_dir(bundle.target_column, bundle.future_dates)
    output_dir.mkdir(parents=True, exist_ok=True)
    row = bundle.model_row

    metric_keys = {
        key: value
        for key, value in row.items()
        if key.startswith("avg_")
        or key.startswith("lead_")
        or key in {
            "calibration",
            "twcrps",
            "mae",
            "rmse",
            "quantile_loss_95",
            "quantile_loss_99",
            "peak_precision",
            "peak_recall",
            "peak_auc",
            "peak_f1",
            "metrics_by_lead_time",
        }
    }

    report = {
        "model_type": row.get("model_type"),
        "experiment_run_id": row.get("run_id"),
        "training_cutoff_date": row.get("training_cutoff_date"),
        "inundation_product": row.get("inundation_product"),
        "target_column": row.get("dataset_target_column", bundle.target_column),
        "target_type": row.get("target_type"),
        "autoregressive": row.get("autoregressive"),
        "seed": row.get("seed"),
        "feature_selection_enabled": row.get("feature_selection_enabled"),
        "selected_feature_count_by_lead": row.get("selected_feature_count_by_lead"),
        "forecast_horizon_periods": row.get("dataset_forecast_horizon_periods"),
        "test_metrics": metric_keys,
        "runtime_retrained_at": row.get("runtime_retrained_at"),
        "runtime_retrained_until_target_date": row.get("runtime_retrained_until_target_date"),
        "runtime_weight_paths_by_lead": row.get("model_weights_path_by_lead"),
        "preprocessing_artifacts": {
            "scaler_path": row.get("dataset_scaler_path"),
            "pca_enabled": row.get("dataset_pca_enabled"),
            "pca_path": row.get("dataset_pca_path"),
        },
    }
    output_path = output_dir / "model_performance.json"
    output_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return output_path


def export_csv(bundle) -> Path:
    output_dir = _prediction_output_dir(bundle.target_column, bundle.future_dates)
    output_dir.mkdir(parents=True, exist_ok=True)

    history_area = convert_target_fraction_to_km2(bundle.history_values_raw, bundle.target_column)
    predicted_area = convert_target_fraction_to_km2(bundle.predicted_raw, bundle.target_column)
    lower_area = convert_target_fraction_to_km2(bundle.lower_raw, bundle.target_column)
    upper_area = convert_target_fraction_to_km2(bundle.upper_raw, bundle.target_column)

    history_df = pd.DataFrame(
        {
            "date": pd.to_datetime(bundle.history_dates),
            "actual_fraction": bundle.history_values_raw,
            "actual_km2": history_area,
            "predicted_fraction": np.nan,
            "predicted_km2": np.nan,
            "lower_bound_80_fraction": np.nan,
            "lower_bound_80_km2": np.nan,
            "upper_bound_80_fraction": np.nan,
            "upper_bound_80_km2": np.nan,
        }
    )
    forecast_df = pd.DataFrame(
        {
            "date": pd.to_datetime(bundle.future_dates),
            "actual_fraction": np.nan,
            "actual_km2": np.nan,
            "predicted_fraction": bundle.predicted_raw,
            "predicted_km2": predicted_area,
            "lower_bound_80_fraction": bundle.lower_raw,
            "lower_bound_80_km2": lower_area,
            "upper_bound_80_fraction": bundle.upper_raw,
            "upper_bound_80_km2": upper_area,
        }
    )
    combined = pd.concat([history_df, forecast_df], ignore_index=True)

    output_path = output_dir / "temporal_predictions.csv"
    combined.to_csv(output_path, index=False)
    return output_path


def update_forecast_performance_csv(bundle) -> Path:
    """Persist forecast records and backfill actual/error columns when available."""

    FORECAST_PERFORMANCE_CSV.parent.mkdir(parents=True, exist_ok=True)

    target_column = str(bundle.target_column)
    predicted_km2 = convert_target_fraction_to_km2(bundle.predicted_raw, target_column)
    lower_km2 = convert_target_fraction_to_km2(bundle.lower_raw, target_column)
    upper_km2 = convert_target_fraction_to_km2(bundle.upper_raw, target_column)

    forecast_time = pd.Timestamp(bundle.origin_date).strftime("%Y-%m-%d")
    model_version = str(bundle.model_row.get("run_id", "unknown"))
    runtime_retrained_at = str(bundle.model_row.get("runtime_retrained_at", "")).strip()
    if runtime_retrained_at:
        model_version = f"{model_version}|{runtime_retrained_at}"

    new_rows = pd.DataFrame(
        {
            "forecast_time": [forecast_time] * len(bundle.future_dates),
            "target_time": [pd.Timestamp(value).strftime("%Y-%m-%d") for value in bundle.future_dates],
            "lead_time": np.arange(1, len(bundle.future_dates) + 1, dtype=int),
            "model_version": [model_version] * len(bundle.future_dates),
            "predicted_fraction": np.asarray(bundle.predicted_raw, dtype=np.float64),
            "predicted_km2": np.asarray(predicted_km2, dtype=np.float64),
            "lower_bound_80_fraction": np.asarray(bundle.lower_raw, dtype=np.float64),
            "lower_bound_80_km2": np.asarray(lower_km2, dtype=np.float64),
            "upper_bound_80_fraction": np.asarray(bundle.upper_raw, dtype=np.float64),
            "upper_bound_80_km2": np.asarray(upper_km2, dtype=np.float64),
            "actual_fraction": np.nan,
            "error_fraction": np.nan,
            "actual_km2": np.nan,
            "error_km2": np.nan,
        }
    )

    expected_columns = [
        "forecast_time",
        "target_time",
        "lead_time",
        "model_version",
        "predicted_fraction",
        "predicted_km2",
        "lower_bound_80_fraction",
        "lower_bound_80_km2",
        "upper_bound_80_fraction",
        "upper_bound_80_km2",
        "actual_fraction",
        "error_fraction",
        "actual_km2",
        "error_km2",
    ]

    if FORECAST_PERFORMANCE_CSV.exists():
        ledger = pd.read_csv(FORECAST_PERFORMANCE_CSV)
    else:
        ledger = pd.DataFrame(columns=expected_columns)

    for column in expected_columns:
        if column not in ledger.columns:
            ledger[column] = np.nan

    ledger = ledger[expected_columns]
    ledger = pd.concat([ledger, new_rows], ignore_index=True)
    ledger = ledger.drop_duplicates(
        subset=["forecast_time", "target_time", "lead_time", "model_version"], keep="last"
    )

    actual_map = {
        pd.Timestamp(date).strftime("%Y-%m-%d"): float(value)
        for date, value in zip(bundle.history_dates, bundle.history_values_raw)
    }
    ledger_target_time = pd.to_datetime(ledger["target_time"], errors="coerce").dt.strftime("%Y-%m-%d")
    mapped_actual = ledger_target_time.map(actual_map)

    known_mask = mapped_actual.notna()
    ledger.loc[known_mask, "actual_fraction"] = mapped_actual.loc[known_mask].astype(np.float64)

    if known_mask.any():
        actual_km2_values = convert_target_fraction_to_km2(
            ledger.loc[known_mask, "actual_fraction"].to_numpy(dtype=np.float64),
            target_column,
        )
        ledger.loc[known_mask, "actual_km2"] = actual_km2_values
        ledger.loc[known_mask, "error_fraction"] = (
            ledger.loc[known_mask, "predicted_fraction"].to_numpy(dtype=np.float64)
            - ledger.loc[known_mask, "actual_fraction"].to_numpy(dtype=np.float64)
        )
        ledger.loc[known_mask, "error_km2"] = (
            ledger.loc[known_mask, "predicted_km2"].to_numpy(dtype=np.float64)
            - ledger.loc[known_mask, "actual_km2"].to_numpy(dtype=np.float64)
        )

    ledger = ledger.sort_values(["forecast_time", "target_time", "lead_time", "model_version"]).reset_index(drop=True)
    ledger.to_csv(FORECAST_PERFORMANCE_CSV, index=False)
    return FORECAST_PERFORMANCE_CSV


def export_graphs(bundle) -> list[Path]:
    output_dir = _prediction_output_dir(bundle.target_column, bundle.future_dates)
    output_dir.mkdir(parents=True, exist_ok=True)

    actual_dates = pd.to_datetime(bundle.history_dates)
    actual_area = convert_target_fraction_to_km2(bundle.history_values_raw, bundle.target_column)
    predicted_area = convert_target_fraction_to_km2(bundle.predicted_raw, bundle.target_column)
    lower_area = convert_target_fraction_to_km2(bundle.lower_raw, bundle.target_column)
    upper_area = convert_target_fraction_to_km2(bundle.upper_raw, bundle.target_column)
    pred_dates = pd.to_datetime(bundle.future_dates)

    title_root = _target_display_name(bundle.target_column)
    paths: list[Path] = []

    windows = [
        (actual_dates.min(), "total_record"),
        (max(actual_dates.min(), actual_dates.max() - pd.Timedelta(days=365 * 5)), "past_five_years"),
        (max(actual_dates.min(), actual_dates.max() - pd.Timedelta(days=365)), "past_year"),
    ]

    for start_date, suffix in windows:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(actual_dates, actual_area, color="#1f77b4", linewidth=1.2, label="Historic VIIRS inundation")

        pred_line_dates = pd.Index([actual_dates.max()]).append(pred_dates)
        pred_line_values = np.concatenate(([actual_area[-1]], predicted_area))
        lower_line = np.concatenate(([actual_area[-1]], lower_area))
        upper_line = np.concatenate(([actual_area[-1]], upper_area))

        ax.plot(pred_line_dates, pred_line_values, color="#d62728", linestyle="--", linewidth=1.4, label="2 month forecast")
        ax.fill_between(
            pred_line_dates,
            lower_line,
            upper_line,
            edgecolor="none",
            color="#d62728",
            alpha=0.18,
            label="80% confidence interval",
        )
        ax.axvline(bundle.origin_date, color="#222", linestyle=":", linewidth=1.0)

        ax.set_title(
            f"{title_root}, {pred_dates.min().date().strftime('%Y-%m-%d')} to {pred_dates.max().date().strftime('%Y-%m-%d')}"
        )
        ax.set_xlabel("Year")
        ax.set_ylabel("Inundated Area (km²)")
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.set_xlim(pd.Timestamp(start_date), pred_dates.max() + pd.Timedelta(days=90))
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        plt.xticks(rotation=45)

        output_path = output_dir / f"prediction_{suffix}.png"
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        paths.append(output_path)

    combined = pd.DataFrame(
        {
            "date": list(actual_dates) + list(pred_dates),
            "km2": list(actual_area) + list(predicted_area),
            "predicted": [False] * len(actual_dates) + [True] * len(pred_dates),
        }
    )
    combined["year"] = combined["date"].dt.year
    # Use a numeric non-leap-year day axis so pandas/matplotlib date converters
    # cannot collapse the repeated calendar-year curves onto one x position.
    combined["plot_day"] = combined["date"].dt.dayofyear
    leap_after_february = combined["date"].dt.is_leap_year & (combined["date"].dt.dayofyear > 59)
    combined.loc[leap_after_february, "plot_day"] -= 1

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.get_cmap("winter")
    unique_years = sorted(combined["year"].unique())
    current_year = pred_dates.max().year
    lower_series = pd.Series(lower_area, index=pred_dates)
    upper_series = pd.Series(upper_area, index=pred_dates)

    for index, year in enumerate(unique_years):
        group = combined[combined["year"] == year]
        if year == current_year:
            actual_group = group[group["predicted"] == False]
            predicted_group = group[group["predicted"] == True]
            if not actual_group.empty:
                ax.plot(
                    actual_group["plot_day"],
                    actual_group["km2"],
                    color="#d62728",
                    linewidth=2.0,
                    label="Actual",
                )
            if not predicted_group.empty:
                anchor_day = None
                anchor_km2 = None
                if not actual_group.empty:
                    anchor_day = actual_group.iloc[-1]["plot_day"]
                    anchor_km2 = float(actual_group.iloc[-1]["km2"])

                pred_days = predicted_group["plot_day"]
                pred_km2 = predicted_group["km2"].to_numpy(dtype=np.float64)
                lower_pred = lower_series.reindex(predicted_group["date"]).to_numpy(dtype=np.float64)
                upper_pred = upper_series.reindex(predicted_group["date"]).to_numpy(dtype=np.float64)

                if anchor_day is not None and anchor_km2 is not None:
                    # Draw a guaranteed connector segment at the handoff point.
                    first_pred_day = pred_days.iloc[0] if len(pred_days) > 0 else None
                    if first_pred_day is None:
                        raise RuntimeError("Predicted group is unexpectedly empty while building the year-by-year plot.")
                    ax.plot(
                        [anchor_day, first_pred_day],
                        [anchor_km2, pred_km2[0]],
                        color="#d62728",
                        linestyle="-",
                        linewidth=2.0,
                        label="_nolegend_",
                    )
                    pred_line_days = pred_days
                    pred_line_km2 = pred_km2
                    fill_days = np.concatenate(([anchor_day], pred_days))
                    fill_lower = np.concatenate(([anchor_km2], lower_pred))
                    fill_upper = np.concatenate(([anchor_km2], upper_pred))
                else:
                    pred_line_days = pred_days
                    pred_line_km2 = pred_km2
                    fill_days = pred_days
                    fill_lower = lower_pred
                    fill_upper = upper_pred

                ax.plot(
                    pred_line_days,
                    pred_line_km2,
                    color="#d62728",
                    linestyle="--",
                    linewidth=2.0,
                    label="2 month forecast",
                )
                ax.fill_between(
                    fill_days,
                    fill_lower,
                    fill_upper,
                    color="#d62728",
                    alpha=0.18,
                    label="80% confidence interval",
                )
        else:
            color = cmap(index / max(len(unique_years) - 1, 1))
            ax.plot(group["plot_day"], group["km2"], color=color, alpha=0.7)

        end_row = group.iloc[-1]
        ax.text(end_row["plot_day"], end_row["km2"], str(year), fontsize=9, va="center")

    ax.set_title(f"{title_root}, Year-by-Year Comparison")
    ax.set_xlabel("Month of Year")
    ax.set_ylabel("Inundated Area (km²)")
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    month_starts = pd.date_range("2001-01-01", "2001-12-01", freq="MS")
    month_days = month_starts.dayofyear.to_numpy()
    ax.set_xlim(1, 365)
    ax.set_xticks(month_days)
    ax.set_xticklabels(month_starts.strftime("%b"))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    comparison_path = output_dir / "prediction_year_by_year_comparison.png"
    fig.savefig(comparison_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    paths.append(comparison_path)

    return paths


def main() -> None:
    try:
        best_row = load_best_model_row()
        update_temporal_sources_for_best_model(best_row)
        runtime_row = retrain_best_model_on_available_data(best_row)
        metadata_path = write_best_model_metadata(runtime_row)
        logging.info("Best temporal model metadata written to %s", metadata_path)
        logging.info(
            "Runtime retraining complete at %s using data through %s",
            runtime_row.get("runtime_retrained_at", "unknown"),
            runtime_row.get("runtime_retrained_until_target_date", "unknown"),
        )

        bundle = generate_temporal_forecast(runtime_row)

        csv_path = export_csv(bundle)
        model_report_path = export_model_report(bundle)
        shap_paths = export_lead_waterfall_plots(
            runtime_row,
            _prediction_output_dir(bundle.target_column, bundle.future_dates),
        )
        performance_path = update_forecast_performance_csv(bundle)
        graph_paths = export_graphs(bundle)
        area_metadata = target_area_conversion_metadata(bundle.target_column)

        logging.info(
            "Temporal predictions exported for %s using %s from run %s.",
            bundle.target_column,
            best_row.get("model_type", "unknown"),
            best_row.get("run_id", "unknown"),
        )
        logging.info("Area conversion metadata: %s", area_metadata)
        logging.info("CSV written to %s", csv_path)
        logging.info("Model performance report written to %s", model_report_path)
        for shap_path in shap_paths:
            logging.info("SHAP explanation written to %s", shap_path)
        logging.info("Forecast performance ledger written to %s", performance_path)
        for graph_path in graph_paths:
            logging.info("Graph written to %s", graph_path)

    except Exception as exc:
        logging.exception("Error occurred while exporting temporal predictions: %s", exc)
        raise


if __name__ == "__main__":
    main()
