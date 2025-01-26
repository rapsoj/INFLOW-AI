"""Code for downloading the Southern Oscillation Index (SOI)."""

import os
from pathlib import Path
from typing import Annotated

from loguru import logger
import requests
import typer

import pandas as pd

SOURCE_URL = "https://www.cpc.ncep.noaa.gov/data/indices/soi"
FILE_PATH_PARTS = ("data/downloads/teleconnections", "soi.txt")
DATA_ROOT = Path(os.getcwd()) # Modify as needed
FOLDER_PATH = os.path.join(DATA_ROOT, 'data/downloads/teleconnections') # Modify as needed

# Dictionary to map month abbreviations to numeric values
MONTH_TO_NUM_UP = {
    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4,
    'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8,
    'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}

def download_soi(
    skip_existing: Annotated[bool, typer.Option(help="Whether to skip an existing file.")] = True,
):
    """Download Southern Oscillation Index data."""
    logger.info("Downloading SOI data...")
    response = requests.get(SOURCE_URL)
    out_file = DATA_ROOT.joinpath(*FILE_PATH_PARTS)
    logger.info(f"Output file path is {out_file}")
    if skip_existing and out_file.exists():
        logger.info("File exists. Skipping.")
    else:
        out_file.parent.mkdir(exist_ok=True, parents=True)
        with out_file.open("w") as fp:
            fp.write(response.text)
        logger.info("Data downloaded to file.")
    logger.success("SOI download complete.")

def read_full_soi_data_anomaly(path: Path | None = None) -> pd.DataFrame:
    """Loads full SOI dataframe. You should use the `read_soi_data` function instead to properly
    subset by time."""
    # Raw data file contains two fixed-width files: first is not standardized, and second is
    # standardized. The standardized values are the most common representation of SOI.
    path = DATA_ROOT / 'data/raw/teleconnections' / 'soi.txt'
    with path.open("r") as fp:
        # Get line number that contains "STANDARDIZED DATA"
        skip_lines = []
        stopping = False
        for line_no, line in enumerate(fp):
            if "STANDARDIZEDDATA" in line.replace(" ", ""):
                stopping = True
                skip_lines.append(line_no-1) # Previous line too
            if stopping:
                skip_lines.append(line_no)
    with path.open("r") as fp:
        for anom_line_no, anom_line in enumerate(fp):
            if "ANOMALY" in anom_line.replace(" ", ""):
                skip_lines.append(anom_line_no-1)
                skip_lines.append(anom_line_no)
                break
    # Skip that line and the next one
    skip_lines.append(anom_line_no+1)
    skip_lines = sorted(skip_lines)
    return pd.read_fwf(path, widths=(4,) + (6,) * 12, skiprows=skip_lines, na_values=("-999.9",))


def read_full_soi_data(path: Path | None = None) -> pd.DataFrame:
    """Loads full SOI dataframe. You should use the `read_soi_data` function instead to properly
    subset by time."""
    # Raw data file contains two fixed-width files: first is not standardized, and second is
    # standardized. The standardized values are the most common representation of SOI.
    path = DATA_ROOT / 'data/raw/teleconnections' / 'soi.txt'
    with path.open("r") as fp:
        # Get line number that contains "STANDARDIZED DATA"
        line_no = 0
        for line_no, line in enumerate(fp):
            if "STANDARDIZEDDATA" in line.replace(" ", ""):
                break
    # Skip that line and the next one
    skiprows = line_no + 2
    return pd.read_fwf(path, widths=(4,) + (6,) * 12, skiprows=skiprows, na_values=("-999.9",))


def import_soi():
    # Import soi 1
    df_soi1 = read_full_soi_data_anomaly(FOLDER_PATH)
    df_soi2 = read_full_soi_data(FOLDER_PATH)
    return df_soi1, df_soi2

def clean_soi1(df_soi1):
    # Clean soi 1
    df_soi1.columns = df_soi1.columns.str.strip()
    df_soi1 = pd.melt(df_soi1, id_vars=['YEAR'], var_name='month', value_name='soi_anom')
    df_soi1 = df_soi1.rename(columns={'YEAR': 'year'})
    df_soi1['month'] = df_soi1['month'].map(MONTH_TO_NUM_UP)
    return df_soi1


def clean_soi2(df_soi2):
    # Clean soi 2
    df_soi2.columns = df_soi2.columns.str.strip()
    df_soi2 = pd.melt(df_soi2, id_vars=['YEAR'], var_name='month', value_name='soi_sd')
    df_soi2 = df_soi2.rename(columns={'YEAR': 'year'})
    df_soi2['month'] = df_soi2['month'].map(MONTH_TO_NUM_UP)
    return df_soi2

def process_soi():
    download_soi()
    df_soi1, df_soi2 = import_soi()
    df_soi1 = clean_soi1(df_soi1)
    df_soi2 = clean_soi2(df_soi2)
    df_soi = pd.merge(df_soi1, df_soi2, on=['year', 'month'])
    return df_soi
