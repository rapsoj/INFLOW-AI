# INFLOW-AI Flood Inundation Prediction Model

## Overview
This program is a comprehensive tool designed to predict flood inundation coverage over the INFLOW study area. It leverages satellite data, machine learning models, and Monte Carlo simulations to generate 2-month predictions and 95% confidence intervals. Additionally, it automates the processing, normalisation, and visualisation of data, providing actionable insights into flood dynamics in the While Nile basin.

---

## Table of Contents
1. [Requirements](#requirements)  
2. [Installation](#installation)  
3. [Usage](#usage)  
4. [Program Workflow](#program-workflow)  
5. [Contributors](#contributors)  

---

## Requirements
- Python 3.11 or higher
- Required Python libraries:
  - `datetime==5.5`
  - `geopandas==1.0.1`
  - `h5py==3.12.1`
  - `loguru==0.7.3`
  - `matplotlib==3.10.0`
  - `netCDF4==1.7.2`
  - `numpy==1.26.4`
  - `pandas==2.2.2`
  - `pathlib==1.0.1`
  - `py_hydroweb==1.0.2`
  - `rasterio==1.4.3`
  - `requests==2.32.3`
  - `scikit-learn==1.6.1`
  - `scipy==1.13.1`
  - `tensorflow==2.17.1`
  - `tqdm==4.67.1`
  - `typer==0.15.1`
  - `xarray==2025.1.1`
- Operating system: Windows, macOS, or Linux

---

## Installation
1. Clone the repository:
'git clone https://github.com/your-repository/flood-inundation-prediction.git`
`cd flood-inundation-prediction`

2. Set up a virtual environment (optional but recommended):
`python3 -m venv env source env/bin/activate # On Windows, use env\Scripts\activate`

3. Install the required libraries:
`pip install -r requirements.txt`


---

## Usage
1. **Data Preparation**: Ensure your input data files (`temporal_data.csv` and `baseline_data.csv`) are correctly formatted and located in the designated folder (`/data` by default).
2. **Run the Program**: Use the following command to start the program:

`python main.py`

3. **Visualization**: After execution, view the generated graphs and reports in the `/output` directory.

---

## Program Workflow
1. **Data Ingestion**: 
- The program reads temporal and baseline data from CSV files.
2. **Data Normalization**: 
- Data is scaled to ensure compatibility with machine learning models.
3. **Model Training**: 
- Trains a regression model on the baseline data.
4. **Monte Carlo Simulation**: 
- Generates predictions and confidence intervals based on 10,000 iterations.
5. **Visualization**: 
- Produces heatmaps and time-series plots of flood inundation predictions.
6. **Output**: 
- Saves prediction data and visualizations in the `/output` directory.


---

## License
This project is licensed under the [MIT License](LICENSE).

---

## Contact
For questions or issues, please contact:
- Email: jessicakristenr@gmail.com
