"""Code for downloading the Pacific Decadal Oscillation (PDO) Index."""

import os
from pathlib import Path
from typing import Annotated

from loguru import logger
import requests
import typer

import numpy as np
import pandas as pd

SOURCE_URL = "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat"
FILE_PATH_PARTS = ("data/raw/teleconnections", "pdo.txt")
DATA_ROOT = Path(os.getcwd()) # Modify as needed
FOLDER_PATH = os.path.join(DATA_ROOT, 'data/raw/teleconnections') # Modify as needed

# Dictionary to map month abbreviations to numeric values
MONTH_TO_NUM = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
    'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
    'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}

def download_pdo(
    skip_existing: Annotated[bool, typer.Option(help="Whether to skip an existing file.")] = True,
):
    """Download Pacific Decadal Oscillation (PDO) Index data."""
    logger.info("Downloading PDO Index data...")
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
    logger.success("PDO Index download complete.")


def import_pdo():
    # Import pdo dataset
    df_pdo = pd.read_table(os.path.join(FOLDER_PATH, "pdo.txt"), delim_whitespace=True, skiprows=1)
    return df_pdo


def clean_pdo(df_pdo):
    # Basic cleaning for pdo dataset
    df_pdo = pd.melt(df_pdo, id_vars=['Year'], var_name='Month', value_name='pdo')
    df_pdo = df_pdo.rename(columns={'Year': 'year', 'Month': 'month'})
    df_pdo['pdo'] = df_pdo['pdo'].replace(99.99, np.nan)  # Remove future values (missing)
    df_pdo['month'] = df_pdo['month'].map(MONTH_TO_NUM)
    return df_pdo

def process_pdo():
    download_pdo()
    df_pdo = import_pdo()
    df_pdo = clean_pdo(df_pdo)
    return df_pdo
