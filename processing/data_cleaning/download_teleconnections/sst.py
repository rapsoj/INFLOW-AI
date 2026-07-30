"""Code for downloading Nino Regions Sea Surface Temperature (SST) indices.
"""
import os
from pathlib import Path
from typing import Annotated

from loguru import logger
import requests
import typer

import pandas as pd

from ...config import get_cfg

SOURCE_URL = get_cfg("sources.teleconnections.sst", "https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices")
FOLDER_PATH = Path(get_cfg("paths.downloads.teleconnections", "data/downloads/teleconnections"))
SST_FILE_PATH = FOLDER_PATH / "sst.txt"

def download_sst(
    skip_existing: Annotated[bool, typer.Option(help="Whether to skip an existing file.")] = False,
):
    """Download Nino Regions Sea Surface Temperatures (SST) data."""
    logger.info("Downloading SST data...")
    response = requests.get(SOURCE_URL)
    out_file = SST_FILE_PATH
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
    df_sst = pd.read_table(SST_FILE_PATH, sep=r"\s+")
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