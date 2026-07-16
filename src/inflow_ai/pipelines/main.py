"""Package entrypoint facade for the legacy top-level workflow."""

from __future__ import annotations

from functools import lru_cache
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType


_PUBLIC_NAMES = {
    "main",
    "get_future_dates",
    "check_if_new_data",
    "update_data",
    "create_dataframe",
    "custom_loss",
    "predict_new_inundation_transformer",
    "predict_new_inundation_rf",
    "monte_carlo_predictions",
    "re_scale_predictions",
    "print_trigger",
    "export_csv",
    "export_graphs",
}


@lru_cache(maxsize=1)
def _load_legacy_entrypoint() -> ModuleType:
    legacy_path = Path(__file__).resolve().parents[3] / "__main__.py"
    spec = spec_from_file_location("inflow_ai._legacy_entrypoint", legacy_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load legacy entrypoint from {legacy_path}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def __getattr__(name: str):
    if name not in _PUBLIC_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    return getattr(_load_legacy_entrypoint(), name)


def __dir__():
    legacy = _load_legacy_entrypoint()
    return sorted(set(globals()) | set(dir(legacy)))


def main(*args, **kwargs):
    return _load_legacy_entrypoint().main(*args, **kwargs)
