from __future__ import annotations

from abc import ABC, abstractmethod

import joblib
import numpy as np


class AblationModel(ABC):
    """Base class that enforces a common interface for ablation models."""

    model_type: str

    def __init__(self, seed: int):
        self.seed = seed
        self.model = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_samples(self, X: np.ndarray) -> np.ndarray:
        """Return predictive samples with shape (n_samples, n_obs)."""
        point_pred = self.predict(X)
        return np.expand_dims(point_pred, axis=0)

    def save_weights(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("Model is not initialized.")
        joblib.dump(self.model, path)
