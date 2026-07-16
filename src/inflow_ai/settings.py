"""Configuration loader for INFLOW-AI."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import os

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


ROOT = Path(__file__).resolve().parents[2]
DEFAULTS_PATH = ROOT / "config" / "defaults.toml"


def _read_toml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


def _join(*parts: str) -> str:
    return str(Path(*parts))


@dataclass(frozen=True)
class Settings:
    data_dir: str
    predictions_dir: str
    models_dir: str
    temporal_sequence_length: int
    temporal_forecast_length: int
    trigger_delta: float
    mc_samples: int
    patch_size: int
    stride: int
    sequence_length: int
    forecast_length: int
    border: int
    temporal_seasonal: str
    historic_dir: str
    stats_dir: str
    maps_dir: str
    downloads_dir: str
    inundation_temporal_unscaled: str
    inundation_temporal_scaled: str
    seasonal_means: str
    seasonal_stds: str
    spatial_model: str
    temporal_model: str
    temporal_model_rf: str
    pca_model: str

    @property
    def inundation_prediction_folder_prefix(self) -> str:
        return "inundation_predictions"

    def prediction_run_dir(self, start_date: str, end_date: str) -> Path:
        return Path(self.predictions_dir) / f"{self.inundation_prediction_folder_prefix}_{start_date}_to_{end_date}"

    def spatial_output_dir(self, folder_title: str) -> Path:
        return Path(self.predictions_dir) / folder_title / "spatial_predictions"

    def explanation_dir(self, folder_title: str) -> Path:
        return Path(self.predictions_dir) / folder_title / "explanations"

    def path(self, value: str) -> str:
        return _join(value)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    defaults = _read_toml(DEFAULTS_PATH)
    project = defaults.get("project", {})
    pipeline = defaults.get("pipeline", {})
    spatial = defaults.get("spatial", {})
    files = defaults.get("files", {})

    data_dir = _env("INFLOW_AI_DATA_DIR", project.get("data_dir", "data"))
    predictions_dir = _env("INFLOW_AI_PREDICTIONS_DIR", project.get("predictions_dir", "predictions"))
    models_dir = _env("INFLOW_AI_MODELS_DIR", project.get("models_dir", "model"))

    return Settings(
        data_dir=data_dir,
        predictions_dir=predictions_dir,
        models_dir=models_dir,
        temporal_sequence_length=_env_int("INFLOW_AI_TEMPORAL_SEQUENCE_LENGTH", pipeline.get("temporal_sequence_length", 36)),
        temporal_forecast_length=_env_int("INFLOW_AI_TEMPORAL_FORECAST_LENGTH", pipeline.get("temporal_forecast_length", 6)),
        trigger_delta=_env_float("INFLOW_AI_TRIGGER_DELTA", pipeline.get("trigger_delta", 0.05)),
        mc_samples=_env_int("INFLOW_AI_MC_SAMPLES", pipeline.get("mc_samples", 1000)),
        patch_size=_env_int("INFLOW_AI_PATCH_SIZE", spatial.get("patch_size", 64)),
        stride=_env_int("INFLOW_AI_STRIDE", spatial.get("stride", 32)),
        sequence_length=_env_int("INFLOW_AI_SEQUENCE_LENGTH", spatial.get("sequence_length", 6)),
        forecast_length=_env_int("INFLOW_AI_FORECAST_LENGTH", spatial.get("forecast_length", 6)),
        border=_env_int("INFLOW_AI_BORDER", spatial.get("border", 4)),
        temporal_seasonal=_env("INFLOW_AI_TEMPORAL_SEASONAL", files.get("temporal_seasonal", _join(data_dir, "temporal_data_seasonal_df.csv"))),
        historic_dir=_env("INFLOW_AI_HISTORIC_DIR", files.get("historic_dir", _join(data_dir, "historic"))),
        stats_dir=_env("INFLOW_AI_STATS_DIR", files.get("stats_dir", _join(data_dir, "stats"))),
        maps_dir=_env("INFLOW_AI_MAPS_DIR", files.get("maps_dir", _join(data_dir, "maps"))),
        downloads_dir=_env("INFLOW_AI_DOWNLOADS_DIR", files.get("downloads_dir", _join(data_dir, "downloads"))),
        inundation_temporal_unscaled=_env("INFLOW_AI_INUNDATION_TEMPORAL_UNSCALED", files.get("inundation_temporal_unscaled", _join(data_dir, "historic", "inundation_temporal_unscaled.csv"))),
        inundation_temporal_scaled=_env("INFLOW_AI_INUNDATION_TEMPORAL_SCALED", files.get("inundation_temporal_scaled", _join(data_dir, "historic", "inundation_temporal_scaled.csv"))),
        seasonal_means=_env("INFLOW_AI_SEASONAL_MEANS", files.get("seasonal_means", _join(data_dir, "stats", "seasonal_means.csv"))),
        seasonal_stds=_env("INFLOW_AI_SEASONAL_STDS", files.get("seasonal_stds", _join(data_dir, "stats", "seasonal_stds.csv"))),
        spatial_model=_env("INFLOW_AI_SPATIAL_MODEL", files.get("spatial_model", _join(models_dir, "spatial_model.keras"))),
        temporal_model=_env("INFLOW_AI_TEMPORAL_MODEL", files.get("temporal_model", _join(models_dir, "temporal_model.keras"))),
        temporal_model_rf=_env("INFLOW_AI_TEMPORAL_MODEL_RF", files.get("temporal_model_rf", _join(models_dir, "temporal_model.pkl"))),
        pca_model=_env("INFLOW_AI_PCA_MODEL", files.get("pca_model", _join(models_dir, "pca_model.pkl"))),
    )