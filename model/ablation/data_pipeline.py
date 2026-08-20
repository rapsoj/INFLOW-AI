from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from processing.config import get_cfg
from .pca_processing import apply_train_only_pca

TargetType = Literal[
    "raw",
    "first_differenced",
    "deseasonalised",
    "seasonally_differenced",
    "differenced_anomaly",
]


@dataclass
class PreparedDataset:
    X_train: np.ndarray
    y_train: np.ndarray
    y_train_raw: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    y_test_raw: np.ndarray
    test_dates: np.ndarray
    train_dates: np.ndarray
    history_dates: np.ndarray
    history_target_raw: np.ndarray
    raw_reconstruction_component_test: np.ndarray
    feature_names: list[str]
    properties: Dict[str, object]


def _seasonal_period(product: str) -> int:
    return 24 if product.lower().strip() == "viirs" else 36


def _dataset_fingerprint(df: pd.DataFrame) -> str:
    payload = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve_aligned_folder(target_product: str) -> Path:
    target_product = target_product.lower().strip()
    configured_root = get_cfg("paths.historic.root", "data/historic")
    return Path(configured_root) / f"{target_product}-aligned"


def _normalize_date_column(df: pd.DataFrame) -> pd.Series:
    for candidate in ("date", "period_start", "measurement_date", "time"):
        if candidate in df.columns:
            return pd.to_datetime(df[candidate], errors="coerce")
    return pd.to_datetime(df.index, errors="coerce")


def _read_temporal_file(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.copy()
    df["date"] = _normalize_date_column(df)
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return df


def _safe_feature_name(name: str, used: set[str], stem: str) -> str:
    if name not in used:
        used.add(name)
        return name

    candidate = f"{stem}_{name}"
    if candidate not in used:
        used.add(candidate)
        return candidate

    i = 2
    while f"{candidate}_{i}" in used:
        i += 1
    resolved = f"{candidate}_{i}"
    used.add(resolved)
    return resolved


def _build_temporal_feature_table(target_product: str) -> tuple[pd.DataFrame, list[str], list[str], str]:
    aligned_dir = _resolve_aligned_folder(target_product)
    if not aligned_dir.exists():
        raise FileNotFoundError(f"Aligned historic directory not found: {aligned_dir}")

    csv_files = sorted(aligned_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No temporal CSV files found in {aligned_dir}")

    merged_dates: pd.DataFrame | None = None
    feature_frames: list[pd.DataFrame] = []
    used_feature_names: set[str] = set()
    rainfall_scaled_cumsum_sources: list[str] = []
    target_column = str(get_cfg("ablation.pipeline.target_column", "percent_inundation")).strip()

    for csv_path in csv_files:
        stem = csv_path.stem
        frame = _read_temporal_file(csv_path)

        if merged_dates is None:
            merged_dates = frame[["date"]].copy()
        else:
            merged_dates = merged_dates.merge(frame[["date"]], on="date", how="outer")

        # Select only numeric temporal columns from each source CSV.
        numeric_cols = [c for c in frame.columns if c != "date" and pd.api.types.is_numeric_dtype(frame[c])]
        if not numeric_cols:
            continue

        source_frame = frame[["date"] + numeric_cols].copy()

        # Extract model target from the product-specific inundation temporal CSV.
        rename_map: dict[str, str] = {}
        for col in numeric_cols:
            if col.startswith("percent_inundation"):
                # Keep inundation columns out of predictors to avoid same-timestep leakage.
                continue

            resolved_name = _safe_feature_name(col, used_feature_names, stem)
            rename_map[col] = resolved_name

        if rename_map:
            source_frame = source_frame[["date"] + list(rename_map.keys())].rename(columns=rename_map)
            feature_frames.append(source_frame)

            # Build the list of columns used for cumulative features after scaling.
            if stem == "rainfall":
                rainfall_scaled_cumsum_sources.extend(rename_map.values())
            if stem == "gridded_rainfall_temporal":
                rainfall_scaled_cumsum_sources.extend(rename_map.values())

    if not any(target_column in frame.columns for frame in [
        _read_temporal_file(p) for p in csv_files if "inundation" in p.stem
    ]):
        raise ValueError(
            f"Could not find configured target column '{target_column}' in aligned folder {aligned_dir}."
        )

    # Re-read target from inundation source to avoid accidental renamed columns.
    if target_product == "viirs":
        target_file_candidates = [
            p for p in csv_files if "inundation" in p.stem and "viirs" in p.stem
        ]
    else:
        target_file_candidates = [
            p for p in csv_files if "inundation" in p.stem and "modis" in p.stem
        ]
    if not target_file_candidates:
        target_file_candidates = [p for p in csv_files if "inundation" in p.stem]

    target_df = _read_temporal_file(target_file_candidates[0])
    if target_column not in target_df.columns:
        raise ValueError(f"Missing configured target column '{target_column}' in {target_file_candidates[0]}")
    target_series = target_df[["date", target_column]].rename(columns={target_column: "target_raw"})

    if merged_dates is None:
        raise RuntimeError("No temporal date index could be built from aligned temporal CSV files.")

    work = merged_dates.merge(target_series, on="date", how="left")
    for frame in feature_frames:
        work = work.merge(frame, on="date", how="left")

    work = work.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return work, sorted(set(rainfall_scaled_cumsum_sources)), [p.name for p in csv_files], str(aligned_dir)


def _target_transform(df: pd.DataFrame, target_type: TargetType, product: str) -> pd.DataFrame:
    season = _seasonal_period(product)
    out = df.copy()

    if target_type == "raw":
        out["target"] = out["target_raw"]
        out["reconstruction_component"] = 0.0
    elif target_type == "first_differenced":
        out["target"] = out["target_raw"].diff(1)
        out["reconstruction_component"] = out["target_raw"].shift(1)
    elif target_type == "deseasonalised":
        month_day = out["date"].dt.strftime("%m-%d")
        seasonal_mean = out.groupby(month_day)["target_raw"].transform("mean")
        out["target"] = out["target_raw"] - seasonal_mean
        out["reconstruction_component"] = seasonal_mean
    elif target_type == "seasonally_differenced":
        seasonal_lag = out["target_raw"].shift(season)
        out["target"] = out["target_raw"] - seasonal_lag
        out["reconstruction_component"] = seasonal_lag
    elif target_type == "differenced_anomaly":
        rolling_mean = out["target_raw"].rolling(window=season, min_periods=3).mean()
        anomaly = out["target_raw"] - rolling_mean
        anomaly_lag = anomaly.shift(1)
        out["target"] = anomaly - anomaly_lag
        out["reconstruction_component"] = anomaly_lag + rolling_mean
    else:
        raise ValueError(f"Unknown target_type='{target_type}'.")

    return out


def _target_lag_features_enabled() -> bool:
    return bool(get_cfg("ablation.pipeline.optional_feature_blocks.include_target_lag_features", True))


def _feature_lag_features_enabled() -> bool:
    return bool(get_cfg("ablation.pipeline.optional_feature_blocks.include_feature_lag_features", False))


def _resolve_forecast_horizon_periods(dates: pd.Series) -> tuple[int, str, float]:
    raw_value = get_cfg("ablation.pipeline.forecast_horizon_periods", "auto_two_months")

    if isinstance(raw_value, (int, np.integer)):
        periods = int(raw_value)
        if periods < 1:
            raise ValueError("ablation.pipeline.forecast_horizon_periods must be >= 1 (no same-day forecast).")
        return periods, "explicit", float("nan")

    if isinstance(raw_value, str):
        normalized = raw_value.strip().lower()
        if normalized in {"auto", "auto_two_months", "2m", "two_months"}:
            months = float(get_cfg("ablation.pipeline.forecast_horizon_months", 2.0))
            unique_dates = pd.Series(pd.to_datetime(dates, errors="coerce")).dropna().drop_duplicates().sort_values()
            diffs = unique_dates.diff().dt.total_seconds().div(86400.0).dropna()
            diffs = diffs[diffs > 0.0]
            if diffs.empty:
                raise RuntimeError("Unable to infer forecast horizon cadence from dates: no positive date differences found.")

            median_period_days = float(np.median(diffs.to_numpy(dtype=np.float64)))
            approx_horizon_days = months * 30.4375
            periods = int(round(approx_horizon_days / median_period_days))
            periods = max(1, periods)
            return periods, "auto_two_months", median_period_days

        try:
            periods = int(normalized)
        except Exception as e:
            raise ValueError(
                "ablation.pipeline.forecast_horizon_periods must be an integer >= 1 or one of "
                "{auto_two_months, auto, 2m, two_months}."
            ) from e
        if periods < 1:
            raise ValueError("ablation.pipeline.forecast_horizon_periods must be >= 1 (no same-day forecast).")
        return periods, "explicit", float("nan")

    raise ValueError(
        "ablation.pipeline.forecast_horizon_periods must be an integer >= 1 or a supported string mode."
    )


def _add_optional_feature_blocks(
    df: pd.DataFrame,
    target_for_lags: pd.Series,
    autoregressive: bool,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    out = df.copy()
    feature_lag_columns: list[str] = []
    target_lag_columns: list[str] = []

    lag_steps = [int(v) for v in get_cfg("ablation.pipeline.optional_feature_blocks.lag_steps", [1, 2, 3, 6])]

    if _feature_lag_features_enabled():
        source_cols = [c for c in out.columns if c != "date" and not c.startswith("lag_") and "_lag_" not in c]
        for lag in lag_steps:
            for col in source_cols:
                lag_col = f"{col}_lag_{lag}"
                out[lag_col] = out[col].shift(lag)
                feature_lag_columns.append(lag_col)

    if get_cfg("ablation.pipeline.optional_feature_blocks.calendar", True):
        out["month"] = out["date"].dt.month
        out["day_of_year"] = out["date"].dt.dayofyear
        out["sin_doy"] = np.sin(2.0 * np.pi * out["day_of_year"] / 366.0)
        out["cos_doy"] = np.cos(2.0 * np.pi * out["day_of_year"] / 366.0)

    if autoregressive and _target_lag_features_enabled():
        for lag in lag_steps:
            out[f"lag_{lag}"] = target_for_lags.shift(int(lag))
            target_lag_columns.append(f"lag_{lag}")

    return out, feature_lag_columns, target_lag_columns


def _fit_train_only_transforms(
    feature_df: pd.DataFrame,
    train_mask: pd.Series,
    scaler_output_path: str,
) -> pd.DataFrame:
    feature_cols = [c for c in feature_df.columns if c != "date"]
    train_features = feature_df.loc[train_mask, feature_cols]

    imputer = SimpleImputer(strategy="median")
    train_imputed = imputer.fit_transform(train_features)

    scaler = StandardScaler()
    scaler.fit(train_imputed)

    full_imputed = imputer.transform(feature_df[feature_cols])
    full_scaled = scaler.transform(full_imputed)

    os.makedirs(os.path.dirname(scaler_output_path), exist_ok=True)
    joblib.dump({"imputer": imputer, "scaler": scaler, "feature_cols": feature_cols}, scaler_output_path)

    scaled_df = pd.DataFrame(full_scaled, columns=feature_cols, index=feature_df.index)
    scaled_df.insert(0, "date", feature_df["date"].values)
    return scaled_df


def _add_post_scale_cumulative_features(scaled_df: pd.DataFrame, source_cols: list[str]) -> pd.DataFrame:
    out = scaled_df.copy()
    for col in source_cols:
        if col in out.columns:
            out[f"{col}_cumulative_scaled"] = out[col].cumsum()
    return out


def reconstruct_raw_from_transformed(
    y_pred_transformed: np.ndarray,
    raw_reconstruction_component: np.ndarray,
) -> np.ndarray:
    """Reconstruct raw-scale predictions from transformed-scale predictions.

    This is intentionally centralized so metric computation is always done
    in raw target space regardless of target_type.
    """
    y_pred = np.asarray(y_pred_transformed, dtype=np.float64)
    component = np.asarray(raw_reconstruction_component, dtype=np.float64)

    if y_pred.ndim == 1:
        if component.shape[0] != y_pred.shape[0]:
            raise ValueError("1D prediction/component length mismatch while reconstructing raw values.")
        return y_pred + component

    if y_pred.ndim == 2:
        if y_pred.shape[1] != component.shape[0]:
            raise ValueError("2D prediction/component shape mismatch while reconstructing raw values.")
        return y_pred + component[np.newaxis, :]

    raise ValueError("Predictions must be 1D or 2D for raw reconstruction.")


def prepare_dataset(
    inundation_product: str,
    target_type: TargetType,
    autoregressive: bool,
    training_cutoff_date: str | None,
    run_id: str | None = None,
    model_type: str | None = None,
    forecast_horizon_periods_override: int | None = None,
) -> PreparedDataset:
    raw_temporal, cumulative_sources, source_files, aligned_dir = _build_temporal_feature_table(inundation_product)
    if forecast_horizon_periods_override is not None:
        forecast_horizon_periods = int(forecast_horizon_periods_override)
        if forecast_horizon_periods < 1:
            raise ValueError("forecast_horizon_periods_override must be >= 1.")
        horizon_mode = "override"
        unique_dates = pd.Series(pd.to_datetime(raw_temporal["date"], errors="coerce")).dropna().drop_duplicates().sort_values()
        diffs = unique_dates.diff().dt.total_seconds().div(86400.0).dropna()
        inferred_period_days = float(np.median(diffs.to_numpy(dtype=np.float64))) if not diffs.empty else float("nan")
    else:
        forecast_horizon_periods, horizon_mode, inferred_period_days = _resolve_forecast_horizon_periods(
            raw_temporal["date"]
        )

    transformed_target = _target_transform(raw_temporal[["date", "target_raw"]], target_type=target_type, product=inundation_product)

    # Feature table starts with all joined temporal predictors except target.
    base_features = raw_temporal.drop(columns=["target_raw"]).copy()

    # Remove known non-feature metadata columns if present.
    for non_feature in ("period_order",):
        if non_feature in base_features.columns and get_cfg("ablation.pipeline.drop_metadata_columns", True):
            base_features = base_features.drop(columns=[non_feature])

    target_lag_toggle = _target_lag_features_enabled()
    target_lag_features_used = bool(autoregressive and target_lag_toggle)

    feature_lag_toggle = _feature_lag_features_enabled()

    features_with_blocks, feature_lag_columns, target_lag_columns = _add_optional_feature_blocks(
        base_features,
        target_for_lags=transformed_target["target"],
        autoregressive=autoregressive,
    )

    work_df = features_with_blocks.merge(
        transformed_target[["date", "target", "reconstruction_component", "target_raw"]],
        on="date",
        how="left",
    )

    # Build explicit forecast labels: features at t predict target at t + horizon.
    work_df["target_date"] = work_df["date"].shift(-forecast_horizon_periods)
    work_df["target"] = work_df["target"].shift(-forecast_horizon_periods)
    work_df["target_raw"] = work_df["target_raw"].shift(-forecast_horizon_periods)
    work_df["reconstruction_component"] = work_df["reconstruction_component"].shift(-forecast_horizon_periods)

    work_df = work_df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    work_df = work_df.dropna(subset=["target", "target_raw", "reconstruction_component", "target_date"])

    if training_cutoff_date:
        cutoff = pd.to_datetime(training_cutoff_date)
        train_mask = work_df["date"] <= cutoff
        test_mask = work_df["date"] > cutoff
    else:
        split_idx = int(0.8 * len(work_df))
        train_mask = pd.Series(False, index=work_df.index)
        train_mask.iloc[:split_idx] = True
        test_mask = ~train_mask

    if int(train_mask.sum()) == 0 or int(test_mask.sum()) == 0:
        raise RuntimeError("Invalid train/test split. Adjust training_cutoff_date or verify source data coverage.")

    artifacts_dir = Path(get_cfg("ablation.artifacts.base_dir", "model/ablation/artifacts"))
    scaler_dir = artifacts_dir / "scalers"
    scaler_file = f"scaler_{inundation_product}_{model_type or 'model'}_{target_type}_{run_id or 'run'}.pkl"
    scaler_path = str(scaler_dir / scaler_file)

    feature_df = work_df.drop(columns=["target", "target_raw", "reconstruction_component", "target_date"])  # includes date
    scaled_features = _fit_train_only_transforms(feature_df, train_mask=train_mask, scaler_output_path=scaler_path)
    scaled_features = _add_post_scale_cumulative_features(scaled_features, cumulative_sources)

    pca_enabled = bool(get_cfg("ablation.pipeline.pca.enabled", False))
    pca_path = ""
    pca_metadata: dict[str, float | int | str] = {}
    if pca_enabled:
        pca_dir = artifacts_dir / "pca"
        pca_file = f"pca_{inundation_product}_{model_type or 'model'}_{target_type}_{run_id or 'run'}.pkl"
        pca_path = str(pca_dir / pca_file)
        pca_n_components = get_cfg("ablation.pipeline.pca.n_components", 0.95)
        pca_whiten = bool(get_cfg("ablation.pipeline.pca.whiten", False))
        pca_seed = int(get_cfg("ablation.pipeline.pca.seed", get_cfg("ablation.experiments.seed", 42)))
        scaled_features, pca_metadata = apply_train_only_pca(
            feature_df=scaled_features,
            train_mask=train_mask,
            output_path=pca_path,
            n_components=pca_n_components,
            whiten=pca_whiten,
            seed=pca_seed,
        )

    feature_cols = [c for c in scaled_features.columns if c != "date"]

    X_train = scaled_features.loc[train_mask, feature_cols].to_numpy(dtype=np.float32)
    X_test = scaled_features.loc[test_mask, feature_cols].to_numpy(dtype=np.float32)

    y_train = work_df.loc[train_mask, "target"].to_numpy(dtype=np.float32)
    y_test = work_df.loc[test_mask, "target"].to_numpy(dtype=np.float32)

    y_train_raw = work_df.loc[train_mask, "target_raw"].to_numpy(dtype=np.float32)
    y_test_raw = work_df.loc[test_mask, "target_raw"].to_numpy(dtype=np.float32)

    test_dates = work_df.loc[test_mask, "target_date"].to_numpy()
    train_dates = work_df.loc[train_mask, "target_date"].to_numpy()
    history_dates = raw_temporal["date"].to_numpy()
    history_target_raw = raw_temporal["target_raw"].to_numpy(dtype=np.float32)
    component = work_df.loc[test_mask, "reconstruction_component"].to_numpy(dtype=np.float32)

    fingerprint_df = scaled_features.merge(
        work_df[["date", "target_raw", "target"]],
        on="date",
        how="left",
    )

    properties = {
        "dataset_source_path": aligned_dir,
        "dataset_target_column": str(get_cfg("ablation.pipeline.target_column", "percent_inundation")),
        "dataset_source_files": "|".join(source_files),
        "dataset_rows_total": int(len(work_df)),
        "dataset_rows_train": int(train_mask.sum()),
        "dataset_rows_test": int(test_mask.sum()),
        "dataset_feature_count": int(len(feature_cols)),
        "dataset_start_date": str(work_df["date"].min().date()),
        "dataset_end_date": str(work_df["date"].max().date()),
        "dataset_target_start_date": str(pd.to_datetime(work_df["target_date"]).min().date()),
        "dataset_target_end_date": str(pd.to_datetime(work_df["target_date"]).max().date()),
        "dataset_missing_ratio": float(raw_temporal.isna().mean(numeric_only=True).mean()),
        "dataset_target_mean": float(work_df["target_raw"].mean()),
        "dataset_target_std": float(work_df["target_raw"].std(ddof=0)),
        "dataset_forecast_horizon_periods": int(forecast_horizon_periods),
        "dataset_forecast_horizon_mode": horizon_mode,
        "dataset_inferred_period_days": inferred_period_days,
        "dataset_forecast_horizon_days_approx": float(forecast_horizon_periods * inferred_period_days)
        if np.isfinite(inferred_period_days)
        else float("nan"),
        "dataset_autoregressive_requested": bool(autoregressive),
        "dataset_target_lag_feature_toggle": bool(target_lag_toggle),
        "dataset_target_lag_features_used": bool(target_lag_features_used),
        "dataset_target_lag_feature_count": int(len(target_lag_columns)),
        "dataset_target_lag_steps": "|".join(
            str(int(v)) for v in get_cfg("ablation.pipeline.optional_feature_blocks.lag_steps", [1, 2, 3, 6])
        )
        if target_lag_features_used
        else "",
        "dataset_feature_lag_feature_toggle": bool(feature_lag_toggle),
        "dataset_feature_lag_features_used": bool(feature_lag_toggle),
        "dataset_feature_lag_feature_count": int(len(feature_lag_columns)),
        "dataset_feature_lag_steps": "|".join(
            str(int(v)) for v in get_cfg("ablation.pipeline.optional_feature_blocks.lag_steps", [1, 2, 3, 6])
        )
        if feature_lag_toggle
        else "",
        "dataset_scaler_path": scaler_path,
        "dataset_pca_enabled": pca_enabled,
        "dataset_pca_path": pca_path,
        "dataset_fingerprint": _dataset_fingerprint(
            fingerprint_df[["date", "target_raw", "target"] + feature_cols].copy()
        ),
    }
    properties.update(pca_metadata)

    return PreparedDataset(
        X_train=X_train,
        y_train=y_train,
        y_train_raw=y_train_raw,
        X_test=X_test,
        y_test=y_test,
        y_test_raw=y_test_raw,
        test_dates=test_dates,
        train_dates=train_dates,
        history_dates=history_dates,
        history_target_raw=history_target_raw,
        raw_reconstruction_component_test=component,
        feature_names=feature_cols,
        properties=properties,
    )
