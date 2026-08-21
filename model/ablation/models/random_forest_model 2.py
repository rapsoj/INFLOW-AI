from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .base_model import AblationModel


class RandomForestAblationModel(AblationModel):
    model_type = "random_forest"

    def __init__(self, seed: int):
        super().__init__(seed=seed)
        self.model = RandomForestRegressor(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_samples(self, X: np.ndarray) -> np.ndarray:
        # Use tree-level predictions as an empirical predictive distribution.
        tree_preds = np.array([tree.predict(X) for tree in self.model.estimators_])
        return tree_preds
