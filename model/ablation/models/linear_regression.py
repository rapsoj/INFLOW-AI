from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression

from .base_model import AblationModel


class LinearRegressionAblationModel(AblationModel):
    model_type = "linear_regression"

    def __init__(self, seed: int):
        super().__init__(seed=seed)
        self.model = LinearRegression()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
