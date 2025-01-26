"""Code for downloading Madden-Julian Oscillation (MJO) pentad indices.
"""

import os
from pathlib import Path
from typing import Annotated

from loguru import logger
import requests
import typer

import pandas as pd

SOURCE_URL = (
    "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_mjo_index/proj_norm_order.ascii"
)
FILE_PATH_PARTS = ("data/downloads/teleconnections", "mjo.txt")
DATA_ROOT = Path(os.getcwd()) # Modify as needed
FOLDER_PATH = os.path.join(DATA_ROOT, 'data/downloads/teleconnections') # Modify as needed


def download_mjo(
    skip_existing: Annotated[bool, typer.Option(help="Whether to skip an existing file.")] = True,
):
    """Download Madden-Julian Oscillation indices."""
    logger.info("Downloading MJO data...")
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
    logger.success("MJO download complete.")


def import_mjo():
    # Import mjo dataset
    df_mjo = pd.read_table(os.path.join(FOLDER_PATH, "mjo.txt"), delim_whitespace=True, skiprows=1)
    return df_mjo


def clean_mjo(df_mjo):
    # Basic cleaning for mjo dataset
    df_mjo = df_mjo.iloc[1:]
    df_mjo.columns = df_mjo.columns.str.strip()
    df_mjo = df_mjo.add_prefix('mjo')
    df_mjo = df_mjo[df_mjo['mjo20E'] != '*****']  # Remove future values (missing)
    # Iterate over columns
    for column_name in df_mjo.columns:
        # Check if the column name contains the substring 'mjo'
        if 'mjo' in column_name:
            # Convert values to float using pd.to_numeric
            df_mjo[column_name] = pd.to_numeric(df_mjo[column_name], errors='coerce')
    df_mjo['year'] = df_mjo['mjoPENTAD'].astype(str).str[:4].astype(int)
    df_mjo['month'] = df_mjo['mjoPENTAD'].astype(str).str[4:6].astype(int)
    df_mjo['day'] = df_mjo['mjoPENTAD'].astype(str).str[6:8].astype(int)

    df_mjo = df_mjo.drop(columns='mjoPENTAD')
    return df_mjo

def process_mjo():
    download_mjo()
    df_mjo = import_mjo()
    df_mjo = clean_mjo(df_mjo)
    return df_mjo
