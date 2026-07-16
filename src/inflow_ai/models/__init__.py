"""Model training and inference package facade."""

from importlib import import_module

__all__ = ["make_spatial_prediction", "train_temporal_model"]


def __getattr__(name):
	if name not in __all__:
		raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

	module = import_module(f"{__name__}.{name}")
	globals()[name] = module
	return module