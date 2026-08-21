from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from data.stats import gridded_stats
from processing import cleaning_utils
from processing.config import get_cfg


def _resolve_target_column(target_column: str | None = None) -> str:
    resolved = str(target_column or get_cfg("ablation.pipeline.target_column", "percent_inundation")).strip()
    if not resolved:
        raise ValueError("Target column cannot be empty for area reconversion.")
    return resolved


def _pixel_area_km2() -> float:
    # Reference transform is defined in projected meters for the aligned VIIRS/MODIS grid.
    a, b, _, d, e, _ = cleaning_utils.MASK_REGIONS_REF_TRANSFORM
    pixel_area_m2 = abs((a * e) - (b * d))
    return float(pixel_area_m2 / 1_000_000.0)


@lru_cache(maxsize=1)
def _region_code_to_pixel_count() -> dict[str, int]:
    regions = cleaning_utils.extract_regions()
    code_map = {v: k for k, v in gridded_stats.region_to_code_dict.items()}

    output: dict[str, int] = {}
    for code, region_name in code_map.items():
        region_data = regions[regions["region"] == region_name]
        if region_data.empty:
            continue
        mask = cleaning_utils.rasterize_to_reference_grid(region_data, all_touched=True, dtype=np.uint8)
        output[code] = int(np.sum(mask == 1))
    return output


def target_denominator_pixels(target_column: str | None = None) -> int:
    resolved = _resolve_target_column(target_column)

    if resolved == "percent_inundation":
        height, width = cleaning_utils.MASK_REGIONS_REF_SHAPE
        return int(height * width)

    if resolved.startswith("percent_inundation_"):
        region_code = resolved.split("percent_inundation_", 1)[1].lower().strip()
        region_pixels = _region_code_to_pixel_count()
        if region_code not in region_pixels:
            raise ValueError(f"Unrecognized inundation target column region code: {region_code}")
        return int(region_pixels[region_code])

    raise ValueError(
        "Area reconversion currently supports 'percent_inundation' and "
        "'percent_inundation_<region_code>' target columns. "
        f"Got '{resolved}'."
    )


def target_area_scale_km2(target_column: str | None = None) -> float:
    return float(target_denominator_pixels(target_column) * _pixel_area_km2())


def convert_target_fraction_to_km2(values: Any, target_column: str | None = None) -> np.ndarray:
    scale = target_area_scale_km2(target_column)
    return np.asarray(values, dtype=np.float64) * scale


def target_area_conversion_metadata(target_column: str | None = None) -> dict[str, float | int | str]:
    resolved = _resolve_target_column(target_column)
    pixel_area_km2 = _pixel_area_km2()
    denominator_pixels = target_denominator_pixels(resolved)
    return {
        "target_column": resolved,
        "pixel_area_km2": float(pixel_area_km2),
        "denominator_pixels": int(denominator_pixels),
        "denominator_area_km2": float(denominator_pixels * pixel_area_km2),
    }
