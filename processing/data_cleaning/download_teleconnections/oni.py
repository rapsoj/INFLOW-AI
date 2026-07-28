"""Code for downloading the Oceanic Nino Index (ONI)."""

import os
from pathlib import Path
from typing import Annotated

from loguru import logger
import requests
import typer

import pandas as pd

from ...config import get_cfg

SOURCE_URL = get_cfg("sources.teleconnections.oni", "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt")
FOLDER_PATH = Path(get_cfg("paths.downloads.teleconnections", "data/downloads/teleconnections"))
ONI_FILE_PATH = FOLDER_PATH / "oni.txt"

def download_oni(
    skip_existing: Annotated[bool, typer.Option(help="Whether to skip an existing file.")] = False,
):
    """Download Oceanic Nino Index data."""
    logger.info("Downloading ONI data...")
    response = requests.get(SOURCE_URL)
    out_file = ONI_FILE_PATH
    logger.info(f"Output file path is {out_file}")
    if skip_existing and out_file.exists():
        logger.info("File exists. Skipping.")
    else:
        out_file.parent.mkdir(exist_ok=True, parents=True)
        with out_file.open("w") as fp:
            fp.write(response.text)
        logger.info("Data downloaded to file.")
    logger.success("ONI download complete.")


def import_oni():
    # Import oni dataset
    df_oni = pd.read_table(ONI_FILE_PATH, sep=r"\s+")
    return df_oni


def clean_oni(df_oni):
    # Basic cleaning for oni dataset
    df_oni = df_oni.rename(columns={'YR': 'year'})
    df_oni = df_oni.rename(columns={c: 'oni' + c for c in df_oni.columns if c not in ['year', 'month']})

    month_conversion_dictionary = {
        'DJF': 1,
        'JFM': 2,
        'FMA': 3,
        'MAM': 4,
        'AMJ': 5,
        'MJJ': 6,
        'JJA': 7,
        'JAS': 8,
        'ASO': 9,
        'SON': 10,
        'OND': 11,
        'NDJ': 12,
    }
    df_oni['month'] = df_oni.oniSEAS.map(month_conversion_dictionary.get)
    return df_oni.drop(columns='oniSEAS')

def process_oni():
    download_oni()
    df_oni = import_oni()
    df_oni = clean_oni(df_oni)
    return df_oni