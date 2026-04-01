import os
import tempfile
import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from pathlib import Path
from rasterio.transform import from_origin

from model.make_spatial_prediction import (
    get_impacted_exposure_points,
    get_exposure_points,
    export_impacted_facilities,
)


def test_get_impacted_exposure_points_basic():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[0, 0] = 1

    # This transform maps raster row 0 to y=9..10 and col 0 to x=0..1
    transform = from_origin(0.0, 10.0, 1.0, 1.0)

    points = gpd.GeoDataFrame(
        {
            "name": ["inside", "outside", "near_edge"],
            "latitude": [9.5, 20.0, 8.5],
            "longitude": [0.5, 0.5, 9.5],
            "facility_type": ["school", "hospital", "school"],
        },
        geometry=gpd.points_from_xy([0.5, 0.5, 9.5], [9.5, 20.0, 8.5]),
        crs="EPSG:4326",
    )

    impacted = get_impacted_exposure_points(points, mask, transform)

    assert impacted.loc[impacted["name"] == "inside", "impacted"].iloc[0] is True
    assert impacted.loc[impacted["name"] == "outside", "impacted"].iloc[0] is False


def test_export_impacted_facilities_creates_csv():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[0, 0] = 1

    metas = {
        "20250211.tif": {
            "transform": from_origin(0.0, 10.0, 1.0, 1.0),
            "crs": "EPSG:4326",
            "height": 10,
            "width": 10,
        }
    }

    points = gpd.GeoDataFrame(
        {
            "name": ["inside", "outside"],
            "latitude": [9.5, 20.0],
            "longitude": [0.5, 0.5],
            "facility_type": ["hospital", "school"],
        },
        geometry=gpd.points_from_xy([0.5, 0.5], [9.5, 20.0]),
        crs="EPSG:4326",
    )

    masks = {
        "Worst Case": mask,
        "Average Case": mask,
        "Best Case": mask,
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir)
        csv_path = export_impacted_facilities(masks, metas, "20250211.tif", points, output_dir)

        assert csv_path is not None
        assert csv_path.exists()

        df = pd.read_csv(csv_path)
        assert "worst_case" in df.columns
        assert "average_case" in df.columns
        assert "best_case" in df.columns

        inside_row = df[df["name"] == "inside"].iloc[0]
        assert inside_row["worst_case"] == True
        assert inside_row["average_case"] == True
        assert inside_row["best_case"] == True

        outside_row = df[df["name"] == "outside"].iloc[0]
        assert outside_row["worst_case"] == False


def test_get_exposure_points_loads_csv_crs():
    with tempfile.TemporaryDirectory() as tmp_dir:
        hospital_csv = Path(tmp_dir) / "hospitals.csv"
        school_csv = Path(tmp_dir) / "schools.csv"

        pd.DataFrame(
            {
                "name": ["test_hospital"],
                "latitude": [9.5],
                "longitude": [30.0],
            }
        ).to_csv(hospital_csv, index=False)

        pd.DataFrame(
            {
                "name": ["test_school"],
                "latitude": [8.5],
                "longitude": [30.1],
            }
        ).to_csv(school_csv, index=False)

        points = get_exposure_points(str(hospital_csv), str(school_csv))

        assert not points.empty
        assert points.crs.to_string() == "EPSG:4326"
        assert set(points["facility_type"]) == {"hospital", "school"}


@pytest.mark.skip("Full run_full_spatial_analysis depends on JASMIN model data and may not be available in CI")
def test_run_full_spatial_analysis_smoke():
    from model.make_spatial_prediction import run_full_spatial_analysis

    # If this runs, it should not raise and should generate the target CSV
    run_full_spatial_analysis()

    # find latest folder
    from model.make_spatial_prediction import load_latest_prediction_csv
    _, folder_title = load_latest_prediction_csv()
    output_csv = Path(f"predictions/{folder_title}/spatial_predictions/impacted_schools_hospitals.csv")
    assert output_csv.exists()
