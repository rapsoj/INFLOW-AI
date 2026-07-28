from __future__ import annotations

import numpy as np
from sklearn.linear_model import ElasticNet

from .base_model import AblationModel


class ElasticNetAblationModel(AblationModel):
    model_type = "elastic_net"

    def __init__(self, seed: int):
        super().__init__(seed=seed)
        self.model = ElasticNet(
            alpha=0.001,
            l1_ratio=0.5,
            max_iter=10000,
            random_state=seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
