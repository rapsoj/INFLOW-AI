"""
Download and extract TAMSAT rainfall estimates.

Please read the API documentation 'TAMSAT_Data_Download_and_Extraction_API.pdf'
before using the API.

The code below provides examples to download and extract TAMSAT
rainfall estimates which you can adapt to your own requirements.
To run the code, copy and paste the code into your Python console.

Author: TAMSAT (tamsat@reading.ac.uk)
"""


# Import the 'download' and 'extract' tools
from tamsat_download_extract_api import download, extract


# Specify local data directory - this must be changed to a suitable
# path on your machine
localdata_dir = '/gws/nopw/j04/tamsat/tamsat/scripts/download_extract_api/data'

# Download TAMSAT files
download({
    "product": 'rfe',
    "timestep": 'dekadal',
    "resolution": 0.25,
    "start_date": '2021-06-01',
    "end_date": '2021-06-30',
    "version": 3.1,
    "localdata_dir": localdata_dir
    })


# Extract TAMSAT rainfall estimates at a point
extract({
    "extract_type": 'point',
    "longitude": 22.73,
    "latitude": -3.51,
    "timestep": 'dekadal',
    "start_date": '2021-06-01',
    "end_date": '2021-06-30',
    "version": 3.1,
    "localdata_dir": localdata_dir
    })
    

# Extract area-average TAMSAT rainfall estimates
extract({
    "extract_type": 'area_average',
    "N": 22.73,
    "S": 20.73,
    "W": 4.0,
    "E": 5.0,
    "timestep": 'dekadal',
    "start_date": '2021-06-01',
    "end_date": '2021-06-30',
    "version": 3.1,
    "localdata_dir": localdata_dir
    })
    

# Subset TAMSAT rainfall estimates for a given domain
extract({
    "extract_type": 'domain',
    "N": 22.73,
    "S": 20.73,
    "W": 4.0,
    "E": 30.0,
    "timestep": 'dekadal',
    "start_date": '2021-06-01',
    "end_date": '2021-06-30',
    "version": '3.1',
    "localdata_dir": localdata_dir
    })
