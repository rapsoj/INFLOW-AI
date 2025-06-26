"""Code for downloading the Oceanic Nino Index (ONI)."""

import os
from pathlib import Path
from typing import Annotated

from loguru import logger
import requests
import typer

import pandas as pd

SOURCE_URL = "https://www.cpc.ncep.noaa.gov/products/international/ocean_monitoring/indian/IODMI/mnth.ersstv5.clim19912020.dmi_current.txt"
FILE_PATH_PARTS = ("data/downloads/teleconnections", "dmi.txt")
DATA_ROOT = Path(os.getcwd()) # Modify as needed
FOLDER_PATH = os.path.join(DATA_ROOT, 'data/downloads/teleconnections') # Modify as needed

def download_dmi(
    skip_existing: Annotated[bool, typer.Option(help="Whether to skip an existing file.")] = False,
):
    """Download Oceanic Nino Index data."""
    logger.info("Downloading DMI data...")
    response = requests.get(SOURCE_URL)
    out_file = DATA_ROOT.joinpath(*FILE_PATH_PARTS)
    logger.info(f"Output file path is {out_file}")
    if skip_existing and out_file.exists():
        logger.info("File exists. Skipping.")
    else:
        out_file.parent.mkdir(exist_ok=True, parents=True)
        with out_file.open("w", encoding="utf-8") as fp:
            fp.write(response.text)
        logger.info("Data downloaded to file.")
    logger.success("DMI download complete.")


def import_dmi():
    # Import dmi dataset
    file_path = os.path.join(FOLDER_PATH, "dmi.txt")
    df_dmi = pd.read_table(file_path, delim_whitespace=True, skiprows=8)
    return df_dmi


def clean_dmi(df_dmi):
    # Basic cleaning for dmi dataset
    df_dmi = df_dmi.rename(columns={
        'Year': 'year', 'Month': 'month', 'WTIO': 'wtio',
        'SETIO': 'setio', 'DMI': 'dmi'})
    return df_dmi

def process_dmi():
    download_dmi()
    df_dmi = import_dmi()
    df_dmi = clean_dmi(df_dmi)
    return df_dmi