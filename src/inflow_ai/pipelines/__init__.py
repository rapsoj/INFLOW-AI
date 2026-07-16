"""Pipeline entrypoints for the package."""

from importlib import import_module

__all__ = ["main"]


def __getattr__(name):
	if name != "main":
		raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

	module = import_module(f"{__name__}.main")
	module_main = module.main
	globals()[name] = module_main
	return module_main