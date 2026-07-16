"""Compatibility facade for the spatial inference module."""

from functools import lru_cache
from importlib import import_module


@lru_cache(maxsize=1)
def _legacy_module():
	return import_module("model.make_spatial_prediction")


def __getattr__(name):
	return getattr(_legacy_module(), name)


def __dir__():
	return sorted(set(globals()) | set(dir(_legacy_module())))