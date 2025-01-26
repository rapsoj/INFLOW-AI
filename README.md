# INFLOW-AI Flood Inundation Prediction Model

## Overview
This program is a comprehensive tool designed to predict flood inundation coverage over the INFLOW study area. It leverages satellite data, machine learning models, and Monte Carlo simulations to generate 2-month predictions and 95% confidence intervals. Additionally, it automates the processing, normalisation, and visualisation of data, providing actionable insights into flood dynamics in the While Nile basin.

![Flooding in White Nile basin](https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExN2RydmRkdng1MzNzMng5YmNlcWhjZjYxc2xydXA0Y3pncWxpeng0aiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/uyXxhhoXumIESZSUi8/giphy.gif)


---

## Table of Contents
1. [Requirements](#requirements)  
2. [Installation](#installation)  
3. [Usage](#usage)  
4. [Program Workflow](#program-workflow)  
5. [Model Description](#model-description)
6. [License](#licence)
7. [Contact](#contact)

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

2. Download the data:
   - Data can be downloaded from the [Google Drive](https://drive.google.com/drive/folders/1TW4Vfhu9SrrVrvonclicaWqy5hHBY7vv?usp=sharing)
   - Save the folder as `data` in the parent directory

4. Set up a virtual environment (optional but recommended):
`python3 -m venv env source env/bin/activate # On Windows, use env\Scripts\activate`

5. Install the required libraries:
`pip install -r requirements.txt`


---

## Usage
1. **Data Preparation**: Ensure your input data files (`temporal_data.csv` and `baseline_data.csv`) are correctly formatted and located in the designated folder (`/data` by default).
2. **Run the Program**: Use the following command to start the program:

`python __main__.py`

3. **Visualisation**: After execution, view the generated graphs and reports in the `/output` directory.

<img src="https://i.imgur.com/m8T8OQW.png" alt="Predictions compared with past year" width="600"/>

<img src="https://i.imgur.com/NUFHPcr.png" alt="Predictions compared with past five years" width="600"/>

---

## Program Workflow
1. **Data Ingestion**: 
- The program reads temporal and baseline data from CSV files.
2. **Data Normalisation**: 
- Data is scaled to ensure compatibility with machine learning models.
3. **Model Training**: 
- Trains a regression model on the baseline data.
4. **Monte Carlo Simulation**: 
- Generates predictions and confidence intervals based on 10,000 iterations.
5. **Visualisation**: 
- Produces heatmaps and time-series plots of flood inundation predictions.
6. **Output**: 
- Saves prediction data and visualisations in the `/output` directory.


---

## Model Description

This model is a **Temporal Transformer-based Recurrent Model** designed for sequence prediction. It leverages the transformer architecture for capturing long-range dependencies and incorporates Monte Carlo Dropout for uncertainty estimation. Below is a detailed breakdown of its components and functionality.

### Model Architecture

- **Input**: The model takes in sequences of temporal data, where each sequence consists of multiple features. The input shape is `(seq_len, num_features)`, where `seq_len` represents the length of the input sequence (e.g., number of time steps), and `num_features` is the number of features at each time step.

- **Positional Encoding**: 
  - The model incorporates **positional encoding** to capture the order of the time steps in the input sequence. This encoding is added to the input data before being fed into the transformer encoder. The positional encoding is generated using a sinusoidal function, which is common in transformer models for sequence processing.

- **Transformer Encoder**: 
  - The core of the model is the **Transformer Encoder**, which processes the sequential input data. The encoder consists of multi-head self-attention layers and feed-forward layers, both equipped with dropout regularisation and layer normalisation.
  - The attention mechanism allows the model to focus on different parts of the sequence when making predictions, and the feed-forward layers learn non-linear relationships.
  
- **Feed-forward Network**:
  - After the transformer encoder, a feed-forward network is applied, consisting of dense layers with ReLU activations. This network helps the model to learn more complex patterns in the data.

- **Output**: 
  - The final output layer produces `predict_ahead` predictions, which represent the forecasted values for the next `predict_ahead` time steps. The output is a dense layer with no activation function (i.e., linear output).

### Loss Function

The model uses a **custom loss function** that combines the Mean Squared Error (MSE) with two additional penalties:
1. **Sign Penalty**: A penalty term that penalises predictions where the sign of the predicted values does not match the sign of the true values. This helps the model to maintain consistency in the directionality of the predictions.
2. **Sum Penalty**: A penalty that ensures the sum of predicted values closely matches the sum of true values across all time steps in the sequence. This is particularly useful in temporal data where the overall trend or aggregate behavior of the series is important.

The final loss is calculated as a weighted combination of the MSE, sign penalty, and sum penalty.

### Training Process

- **Data Preparation**: The data is preprocessed into overlapping sequences with a specified `look_back` window (past observations) and `predict_ahead` (future steps to predict).
  
- **Monte Carlo Dropout**: The model employs **Monte Carlo Dropout** during inference to estimate uncertainty in the predictions. Dropout is kept active during prediction to generate multiple stochastic predictions, from which the mean and standard deviation are computed to capture the uncertainty in the model’s forecasts.

- **Early Stopping**: During training, **early stopping** is used to prevent overfitting by monitoring the validation loss and stopping training when the performance plateaus for a specified number of epochs.

- **Optimisation**: The model is compiled using the **Adam optimiser** with a learning rate of 0.0001, and the model is trained to minimise the custom loss function.

### Model Evaluation

- After training, the model is evaluated using the **mean squared error (MSE)** and **mean absolute error (MAE)** metrics on the test data.
- The model’s predictions are compared against the actual values, and the performance metrics are printed for analysis.

### Model Saving and Loading

- The trained model can be saved for later use using the `.save()` method and can be loaded back using the `load_model()` function from TensorFlow/Keras for inference.

### Key Features:
- Transformer-based architecture with multi-head attention.
- Monte Carlo Dropout for uncertainty estimation.
- Custom loss function with penalties for sign consistency and sum preservation.
- Early stopping to prevent overfitting.
- Model evaluation using MSE and MAE.

---

## License
This project is licensed under the [MIT License](LICENSE).

---

## Contact
For questions or issues, please contact:
- Email: jessicakristenr@gmail.com
