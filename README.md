# INFLOW-AI Flood Inundation Prediction Model

## Overview
This program is a comprehensive tool designed to predict flood inundation percentages for the INFLOW study area. It leverages temporal data, machine learning models, and Monte Carlo simulations to generate future predictions and confidence intervals. Additionally, it automates the processing, normalization, and visualization of data, providing actionable insights into flood dynamics.

---

## Table of Contents
1. [Requirements](#requirements)  
2. [Installation](#installation)  
3. [Usage](#usage)  
4. [Program Workflow](#program-workflow)  
5. [Contributors](#contributors)  

---

## Requirements
- Python 3.8 or higher
- Required Python libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `scipy`
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

## Contributors
- **Your Name**: Project lead and developer.  
- **Other Contributor Names**: Specify roles or contributions.

---

## License
This project is licensed under the [MIT License](LICENSE).

---

## Contact
For questions or issues, please contact:
- Email: jessicakristenr@gmail.com
