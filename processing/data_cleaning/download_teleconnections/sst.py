"""Code for downloading Nino Regions Sea Surface Temperature (SST) indices.
"""
import os
from pathlib import Path
from typing import Annotated

from loguru import logger
import requests
import typer

import pandas as pd

SOURCE_URL = "https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices"
FILE_PATH_PARTS = ("data/downloads/teleconnections", "sst.txt")
DATA_ROOT = Path(os.getcwd()) # Modify as needed
FOLDER_PATH = os.path.join(DATA_ROOT, 'data/downloads/teleconnections') # Modify as needed

def download_sst(
    skip_existing: Annotated[bool, typer.Option(help="Whether to skip an existing file.")] = False,
):
    """Download Nino Regions Sea Surface Temperatures (SST) data."""
    logger.info("Downloading SST data...")
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
    logger.success("Nino Regions SST download complete.")


def import_sst():
    # Import sst dataset
    df_sst = pd.read_table(os.path.join(FOLDER_PATH, "sst.txt"), delim_whitespace=True)
    return df_sst


def clean_sst(df_sst):
    # Basic cleaning for sst dataset
    df_sst = df_sst.rename(columns={'YR': 'year', 'MON': 'month'})
    df_sst = df_sst.rename(columns={c: 'nino' + c for c in df_sst.columns if c not in ['year', 'month']})
    return df_sst

def process_sst():
    download_sst()
    df_sst = import_sst()
    df_sst = clean_sst(df_sst)
    return df_sst