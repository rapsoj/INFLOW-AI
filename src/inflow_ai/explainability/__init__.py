"""Explainability and plotting package facade."""

from importlib import import_module

__all__ = ["plot_explanations"]


def __getattr__(name):
	if name != "plot_explanations":
		raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

	module = import_module(f"{__name__}.plot_explanations")
	globals()[name] = module
	return module