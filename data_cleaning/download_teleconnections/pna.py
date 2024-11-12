"""Code for downloading the Pacific-North American (PNA) pattern index."""

import os
from pathlib import Path
from typing import Annotated

from loguru import logger
import requests
import typer

import pandas as pd

SOURCE_URL = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.pna.monthly.b5001.current.ascii.table"
FILE_PATH_PARTS = ("data/raw/teleconnections", "pna.txt")
DATA_ROOT = Path(os.getcwd()) # Modify as needed
FOLDER_PATH = os.path.join(DATA_ROOT, 'data/raw/teleconnections') # Modify as needed

# Dictionary to map month abbreviations to numeric values
MONTH_TO_NUM = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
    'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
    'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}

def download_pna(
    skip_existing: Annotated[bool, typer.Option(help="Whether to skip an existing file.")] = True,
):
    """Download Pacific-North American Index data."""
    logger.info("Downloading PNA Index data...")
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
    logger.success("PNA Index download complete.")


def import_pna():
    # Import pna dataset
    df_pna = pd.read_table(os.path.join(FOLDER_PATH, "pna.txt"), delim_whitespace=True)
    return df_pna


def clean_pna(df_pna):
    # Basic cleaning for pna dataset
    df_pna['year'] = df_pna.index
    df_pna = df_pna.reset_index(drop=True)
    df_pna = pd.melt(df_pna, id_vars=['year'], var_name='month', value_name='pna')
    df_pna['month'] = df_pna['month'].map(MONTH_TO_NUM)
    return df_pna

def process_pna():
    download_pna()
    df_pna = import_pna()
    df_pna = clean_pna(df_pna)
    return df_pna
