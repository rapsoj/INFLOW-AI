from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from .base_model import AblationModel


class GradientBoostingAblationModel(AblationModel):
    model_type = "gradient_boosting"

    def __init__(self, seed: int):
        super().__init__(seed=seed)
        self.model = GradientBoostingRegressor(
            random_state=seed,
            n_estimators=400,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.9,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
