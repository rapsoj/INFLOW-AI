from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from model.ablation.deployment import (
    _add_optional_feature_blocks,
    _add_post_scale_cumulative_features,
    _apply_saved_feature_transforms,
    _parse_feature_names_by_lead,
    _parse_weights_by_lead,
    _restore_model_wrapper,
    _sync_runtime_config_from_row,
)
from model.ablation.data_pipeline import _build_temporal_feature_table, _target_transform
from model.ablation.area_reconversion import target_area_scale_km2


def export_lead_waterfall_plots(row: dict[str, Any], output_dir: str | Path) -> list[Path]:
    """Export SHAP waterfall plots for each deployed forecast lead.

    The feature table, saved train-only transforms, selected feature names, and
    runtime-retrained weights are the same objects used by deployment inference.
    """
    _sync_runtime_config_from_row(row)
    output_path = Path(output_dir) / "explanations"
    output_path.mkdir(parents=True, exist_ok=True)

    product = str(row["inundation_product"]).strip().lower()
    target_type = str(row["target_type"]).strip()
    target_column = str(row.get("dataset_target_column") or row.get("target_column") or "percent_inundation")
    area_scale_km2 = target_area_scale_km2(target_column)
    fraction_to_percent = 100.0 / area_scale_km2
    autoregressive = str(row.get("autoregressive", False)).strip().lower() in {"1", "1.0", "true", "yes", "y"}
    seed = int(float(row.get("seed", 42)))

    raw_temporal, cumulative_sources, _, _ = _build_temporal_feature_table(product)
    raw_temporal = raw_temporal.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    observed_target_history = raw_temporal.dropna(subset=["target_raw"])
    if observed_target_history.empty:
        raise RuntimeError("Cannot explain predictions without observed target history.")

    target_anchor_date = pd.to_datetime(observed_target_history["date"]).max()
    transformed_target = _target_transform(
        raw_temporal[["date", "target_raw"]].copy(),
        target_type=target_type,
        product=product,
    )
    base_features = raw_temporal.drop(columns=["target_raw"]).copy()
    if "period_order" in base_features.columns:
        from processing.config import get_cfg
        if bool(get_cfg("ablation.pipeline.drop_metadata_columns", True)):
            base_features = base_features.drop(columns=["period_order"])

    features_with_blocks, _, _ = _add_optional_feature_blocks(
        base_features,
        target_for_lags=transformed_target["target"],
        autoregressive=autoregressive,
    )
    transformed_features = _apply_saved_feature_transforms(features_with_blocks, cumulative_sources, row)
    eligible_features = transformed_features[
        pd.to_datetime(transformed_features["date"]) <= target_anchor_date
    ]
    if eligible_features.empty:
        raise RuntimeError("Cannot find an explanation feature row on or before the target date.")
    latest_features = eligible_features.iloc[-1]

    selected_by_lead = _parse_feature_names_by_lead(row.get("selected_feature_names_by_lead"))
    weights_by_lead = _parse_weights_by_lead(row.get("model_weights_path_by_lead"))
    lead_count = int(float(row.get("dataset_forecast_horizon_periods", row.get("dataset_lead_period_max", 1))))
    generated: list[Path] = []
    contribution_percentages: dict[str, list[dict[str, float | str]]] = {}
    history_raw = observed_target_history["target_raw"].to_numpy(dtype=np.float64).tolist()
    season = 24 if product == "viirs" else 36

    for lead in range(1, lead_count + 1):
        selected_features = selected_by_lead.get(lead, [])
        weight_path = weights_by_lead.get(lead)
        if not selected_features or not weight_path:
            raise RuntimeError(f"Missing selected features or runtime weights for lead {lead}.")

        missing = [name for name in selected_features if name not in transformed_features.columns]
        if missing:
            raise RuntimeError(f"Missing SHAP features for lead {lead}: {missing[:10]}")

        model_wrapper = _restore_model_wrapper(
            str(row.get("model_type") or row.get("model_architecture")),
            seed,
            weight_path,
        )
        model = model_wrapper.model
        x_latest = latest_features[selected_features].to_numpy(dtype=np.float64).reshape(1, -1)
        background = transformed_features[selected_features].tail(min(100, len(transformed_features))).to_numpy(dtype=np.float64)

        if model_wrapper.model_type == "random_forest":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(x_latest, check_additivity=False)
        else:
            explainer = shap.Explainer(model.predict, background, feature_names=selected_features)
            shap_values = explainer(x_latest)

        explanation = shap_values[0]
        transformed_prediction = float(np.asarray(model_wrapper.predict(x_latest)).reshape(-1)[0])
        if target_type == "seasonally_differenced":
            lag_index = len(history_raw) - season
            if lag_index < 0:
                raise RuntimeError(f"Not enough history to convert lead {lead} SHAP values to raw km2.")
            reconstruction_component = history_raw[lag_index]
        elif target_type == "deseasonalised":
            month_day = target_anchor_date.strftime("%m-%d")
            seasonal_values = observed_target_history.loc[
                observed_target_history["date"].dt.strftime("%m-%d") == month_day,
                "target_raw",
            ]
            reconstruction_component = float(seasonal_values.mean()) if not seasonal_values.empty else 0.0
        else:
            reconstruction_component = 0.0

        raw_prediction = transformed_prediction + reconstruction_component
        history_raw.append(raw_prediction)
        raw_values = np.asarray(explanation.values, dtype=np.float64) * area_scale_km2 * fraction_to_percent
        absolute_total = float(np.abs(raw_values).sum())
        if absolute_total > 0:
            contribution_shares = np.abs(raw_values) / absolute_total * 100.0
        else:
            contribution_shares = np.zeros_like(raw_values)
        signed_contribution_shares = np.sign(raw_values) * contribution_shares
        contribution_percentages[str(lead)] = [
            {"feature": name, "absolute_shap_share_percent": float(share)}
            for name, share in zip(selected_features, contribution_shares)
        ]
        expected_value = np.asarray(explanation.base_values, dtype=np.float64)
        if expected_value.ndim > 0:
            expected_value = expected_value.reshape(-1)[0]
        raw_base_value = (float(expected_value) + reconstruction_component) * area_scale_km2 * fraction_to_percent
        explanation = shap.Explanation(
            values=signed_contribution_shares,
            base_values=raw_base_value,
            data=x_latest[0],
            feature_names=selected_features,
        )
        plt.figure(figsize=(11, 8))
        shap.plots.waterfall(explanation, max_display=min(20, len(selected_features)), show=False)
        plt.title(
            f"Lead {lead} SHAP drivers ({product}, {target_type}) - "
            "relative contribution (%; +/- direction, absolute shares sum to 100%)"
        )
        output_file = output_path / f"shap_waterfall_lead_{lead}.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=220, bbox_inches="tight")
        plt.close()
        generated.append(output_file)

    metadata = {
        "model_type": row.get("model_type") or row.get("model_architecture"),
        "run_id": row.get("run_id"),
        "target_type": target_type,
        "target_product": product,
        "explanation_date": str(target_anchor_date.date()),
        "lead_count": lead_count,
        "contribution_percentages": contribution_percentages,
        "plots": [str(path) for path in generated],
    }
    (output_path / "shap_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return generated
