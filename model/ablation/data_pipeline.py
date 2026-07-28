from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Dict, Literal

import numpy as np
import pandas as pd

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
    raw_reconstruction_component_test: np.ndarray
    properties: Dict[str, object]


def _candidate_paths(product: str) -> list[str]:
    product = product.lower().strip()
    if product == "viirs":
        return [
            "data/historic/inundation_viirs_temporal.csv",
            "data/historic/viirs_inundation_temporal.csv",
        ]
    return [
        "data/historic/inundation_temporal.csv",
        "data/historic/inundation_modis_temporal.csv",
    ]


def _resolve_source_path(product: str) -> str:
    for path in _candidate_paths(product):
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"No temporal inundation CSV found for product='{product}'. "
        f"Tried: {_candidate_paths(product)}"
    )


def _load_temporal_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "date" in df.columns:
        date_series = pd.to_datetime(df["date"], errors="coerce")
    elif "period_start" in df.columns:
        date_series = pd.to_datetime(df["period_start"], errors="coerce")
    else:
        date_series = pd.to_datetime(df.index, errors="coerce")

    target_col = "percent_inundation"
    if target_col not in df.columns:
        inundation_cols = [c for c in df.columns if c.startswith("percent_inundation")]
        if not inundation_cols:
            raise ValueError(f"No percent inundation column found in '{path}'.")
        target_col = inundation_cols[0]

    out = pd.DataFrame({
        "date": date_series,
        "target_raw": pd.to_numeric(df[target_col], errors="coerce"),
    }).dropna(subset=["date", "target_raw"])
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return out


def _seasonal_period(product: str) -> int:
    return 24 if product.lower().strip() == "viirs" else 36


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


def _build_features(df: pd.DataFrame, autoregressive: bool) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["month"] = df["date"].dt.month
    out["day_of_year"] = df["date"].dt.dayofyear
    out["sin_doy"] = np.sin(2.0 * np.pi * out["day_of_year"] / 366.0)
    out["cos_doy"] = np.cos(2.0 * np.pi * out["day_of_year"] / 366.0)

    if autoregressive:
        for lag in (1, 2, 3, 6):
            out[f"lag_{lag}"] = df["target"].shift(lag)

    return out


def _dataset_fingerprint(df: pd.DataFrame) -> str:
    payload = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prepare_dataset(
    inundation_product: str,
    target_type: TargetType,
    autoregressive: bool,
    training_cutoff_date: str | None,
) -> PreparedDataset:
    source_path = _resolve_source_path(inundation_product)
    raw_df = _load_temporal_df(source_path)
    transformed_df = _target_transform(raw_df, target_type=target_type, product=inundation_product)

    features_df = _build_features(transformed_df, autoregressive=autoregressive)
    work_df = pd.concat([transformed_df, features_df], axis=1)
    work_df = work_df.dropna().reset_index(drop=True)

    feature_cols = [c for c in features_df.columns]
    if not feature_cols:
        raise RuntimeError("No features were generated for the ablation dataset.")

    if training_cutoff_date:
        cutoff = pd.to_datetime(training_cutoff_date)
        train_df = work_df[work_df["date"] <= cutoff]
        test_df = work_df[work_df["date"] > cutoff]
    else:
        split_idx = int(0.8 * len(work_df))
        train_df = work_df.iloc[:split_idx]
        test_df = work_df.iloc[split_idx:]

    if train_df.empty or test_df.empty:
        raise RuntimeError(
            "Invalid train/test split. Adjust training_cutoff_date or verify source data coverage."
        )

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df["target"].to_numpy(dtype=np.float32)
    y_train_raw = train_df["target_raw"].to_numpy(dtype=np.float32)

    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_test = test_df["target"].to_numpy(dtype=np.float32)
    y_test_raw = test_df["target_raw"].to_numpy(dtype=np.float32)
    component = test_df["reconstruction_component"].to_numpy(dtype=np.float32)

    properties = {
        "dataset_source_path": source_path,
        "dataset_rows_total": int(len(work_df)),
        "dataset_rows_train": int(len(train_df)),
        "dataset_rows_test": int(len(test_df)),
        "dataset_feature_count": int(len(feature_cols)),
        "dataset_start_date": str(work_df["date"].min().date()),
        "dataset_end_date": str(work_df["date"].max().date()),
        "dataset_missing_ratio": float(raw_df["target_raw"].isna().mean()),
        "dataset_target_mean": float(work_df["target_raw"].mean()),
        "dataset_target_std": float(work_df["target_raw"].std(ddof=0)),
        "dataset_fingerprint": _dataset_fingerprint(work_df[["date", "target_raw", "target"] + feature_cols]),
    }

    return PreparedDataset(
        X_train=X_train,
        y_train=y_train,
        y_train_raw=y_train_raw,
        X_test=X_test,
        y_test=y_test,
        y_test_raw=y_test_raw,
        raw_reconstruction_component_test=component,
        properties=properties,
    )
