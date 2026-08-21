from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from model.ablation.data_pipeline import _build_temporal_feature_table
from processing.config import get_cfg


def _calendar_day_axis(dates: pd.Series) -> np.ndarray:
    dates = pd.to_datetime(dates)
    day_of_year = dates.dt.dayofyear.to_numpy(dtype=float)
    leap_after_february = dates.dt.is_leap_year.to_numpy() & (day_of_year > 59)
    day_of_year[leap_after_february] -= 1
    return day_of_year


def _plot_year_by_year(series: pd.Series, feature_name: str, output_path: Path) -> None:
    frame = pd.DataFrame({"date": pd.to_datetime(series.index), "value": series.to_numpy(dtype=float)})
    frame = frame.dropna(subset=["date", "value"]).sort_values("date")
    if frame.empty:
        return

    frame["year"] = frame["date"].dt.year
    frame["plot_day"] = _calendar_day_axis(frame["date"])

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.get_cmap("winter")
    years = sorted(frame["year"].unique())
    current_year = max(years)

    for index, (year, group) in enumerate(frame.groupby("year")):
        group = group.sort_values("date")
        color = "#d62728" if year == current_year else cmap(index / max(len(years) - 1, 1))
        ax.plot(
            group["plot_day"],
            group["value"],
            color=color,
            linewidth=2.0 if year == current_year else 1.0,
            alpha=1.0 if year == current_year else 0.7,
            label=str(year),
        )
        ax.text(
            group["plot_day"].iloc[-1],
            group["value"].iloc[-1],
            str(year),
            color=color,
            fontsize=9,
            va="center",
        )

    month_starts = pd.date_range("2001-01-01", "2001-12-01", freq="MS")
    ax.set_xlim(1, 365)
    ax.set_xticks(month_starts.dayofyear.to_numpy())
    ax.set_xticklabels(month_starts.strftime("%b"))
    ax.set_xlabel("Month of Year")
    ax.set_ylabel(feature_name)
    ax.set_title(f"{feature_name}, Year-by-Year Comparison")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def export_feature_history_plots(row: dict[str, Any], output_dir: str | Path) -> list[Path]:
    """Plot every processed temporal predictor in the current prediction folder."""
    product = str(row["inundation_product"]).strip().lower()
    raw_temporal, _, _, _ = _build_temporal_feature_table(product)
    raw_temporal = raw_temporal.sort_values("date").drop_duplicates("date", keep="last")
    raw_temporal = raw_temporal.set_index(pd.to_datetime(raw_temporal["date"])).drop(columns=["date"])

    feature_history_dir = Path(output_dir) / "feature_history"
    feature_history_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    predictor_names = [
        column
        for column in raw_temporal.select_dtypes(include=[np.number]).columns
        if column not in {"target_raw", "year", "month", "period_order"}
    ]
    for feature_name in tqdm(predictor_names, desc="Plotting feature histories"):
        safe_name = feature_name.replace("/", "_").replace(" ", "_")
        output_path = feature_history_dir / f"{safe_name}_year_by_year_comparison.png"
        _plot_year_by_year(raw_temporal[feature_name], feature_name, output_path)
        if output_path.exists():
            generated.append(output_path)

    metadata = {
        "target_product": product,
        "feature_count": len(generated),
        "features": predictor_names,
        "output_directory": str(feature_history_dir),
    }
    pd.Series(metadata, dtype=object).to_json(feature_history_dir / "feature_history_metadata.json")
    return generated
