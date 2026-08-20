from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def apply_train_only_pca(
    feature_df: pd.DataFrame,
    train_mask: pd.Series,
    output_path: str,
    n_components: int | float | str,
    whiten: bool,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """Fit PCA on train rows only, transform full data, and persist the fitted object."""
    feature_cols = [c for c in feature_df.columns if c != "date"]
    train_matrix = feature_df.loc[train_mask, feature_cols].to_numpy(dtype=np.float64)
    full_matrix = feature_df[feature_cols].to_numpy(dtype=np.float64)

    resolved_components: int | float
    if isinstance(n_components, str):
        stripped = n_components.strip().lower()
        if stripped == "mle":
            resolved_components = "mle"  # type: ignore[assignment]
        else:
            resolved_components = float(n_components)
    else:
        resolved_components = n_components

    pca = PCA(
        n_components=resolved_components,
        whiten=whiten,
        svd_solver="auto",
        random_state=seed,
    )
    pca.fit(train_matrix)

    transformed = pca.transform(full_matrix)
    component_cols = [f"pca_{i + 1}" for i in range(transformed.shape[1])]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump({"pca": pca, "feature_cols": feature_cols}, output_path)

    out_df = pd.DataFrame(transformed, columns=component_cols, index=feature_df.index)
    out_df.insert(0, "date", feature_df["date"].values)

    metadata = {
        "pca_components_kept": int(transformed.shape[1]),
        "pca_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
    }
    return out_df, metadata
