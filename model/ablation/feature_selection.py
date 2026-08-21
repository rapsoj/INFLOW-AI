from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from processing.config import get_cfg


@dataclass
class FeatureSelectionResult:
    X_train: np.ndarray
    X_test: np.ndarray
    selected_feature_indices: np.ndarray
    selected_feature_names: list[str]
    removed_constant_features: list[str]
    removed_correlated_features: list[str]
    cv_mean_mae_by_k: dict[int, float]
    best_k: int
    used_cache: bool
    cache_path: str
    enabled: bool


def _redundancy_prune(
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    near_constant_variance_threshold: float,
    high_correlation_threshold: float,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[str], np.ndarray]:
    if X_train.shape[1] != len(feature_names):
        raise ValueError("Feature name count does not match X_train width.")

    variances = np.var(X_train, axis=0)
    keep_variance_mask = variances > near_constant_variance_threshold
    kept_after_variance = [name for name, keep in zip(feature_names, keep_variance_mask) if keep]
    removed_constant = [name for name, keep in zip(feature_names, keep_variance_mask) if not keep]

    if not np.any(keep_variance_mask):
        # Preserve at least one feature to avoid empty design matrices.
        keep_variance_mask[np.argmax(variances)] = True
        kept_after_variance = [name for name, keep in zip(feature_names, keep_variance_mask) if keep]
        removed_constant = [name for name, keep in zip(feature_names, keep_variance_mask) if not keep]

    X_train_var = X_train[:, keep_variance_mask]
    X_test_var = X_test[:, keep_variance_mask]

    n_kept = X_train_var.shape[1]
    if n_kept <= 1:
        keep_corr_mask = np.ones(n_kept, dtype=bool)
        removed_corr: list[str] = []
    else:
        corr = np.corrcoef(X_train_var, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)

        keep_corr_mask = np.ones(n_kept, dtype=bool)
        removed_corr = []
        for i in range(n_kept):
            if not keep_corr_mask[i]:
                continue
            for j in range(i + 1, n_kept):
                if keep_corr_mask[j] and abs(corr[i, j]) >= high_correlation_threshold:
                    keep_corr_mask[j] = False
                    removed_corr.append(kept_after_variance[j])

    pruned_feature_names = [name for name, keep in zip(kept_after_variance, keep_corr_mask) if keep]
    X_train_pruned = X_train_var[:, keep_corr_mask]
    X_test_pruned = X_test_var[:, keep_corr_mask]

    # Indices in the original feature space, after redundancy pruning.
    kept_original_indices = np.where(keep_variance_mask)[0][keep_corr_mask]
    return (
        X_train_pruned,
        X_test_pruned,
        pruned_feature_names,
        removed_constant,
        removed_corr,
        kept_original_indices,
    )


def _rank_features_with_elastic_net(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    alpha: float,
    l1_ratio: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray]:
    estimator = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        max_iter=max_iter,
        random_state=seed,
    )
    estimator.fit(X, y)
    scores = np.abs(estimator.coef_)
    order = np.argsort(scores)[::-1]
    return order, scores


def _choose_k_indices(order: np.ndarray, scores: np.ndarray, k: int) -> np.ndarray:
    nonzero_order = order[scores[order] > 0.0]
    if len(nonzero_order) >= k:
        return nonzero_order[:k]
    return order[:k]


def _cv_mean_mae_for_k(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str,
    seed: int,
    k: int,
    cv_splits: int,
    en_alpha: float,
    en_l1_ratio: float,
    en_max_iter: int,
) -> float:
    from .models import MODEL_REGISTRY

    tscv = TimeSeriesSplit(n_splits=cv_splits)
    fold_losses: list[float] = []

    for fold_train_idx, fold_val_idx in tscv.split(X_train):
        X_fold_train = X_train[fold_train_idx]
        y_fold_train = y_train[fold_train_idx]
        X_fold_val = X_train[fold_val_idx]
        y_fold_val = y_train[fold_val_idx]

        feature_order, feature_scores = _rank_features_with_elastic_net(
            X=X_fold_train,
            y=y_fold_train,
            seed=seed,
            alpha=en_alpha,
            l1_ratio=en_l1_ratio,
            max_iter=en_max_iter,
        )
        selected = _choose_k_indices(feature_order, feature_scores, k)

        model = MODEL_REGISTRY[model_type](seed=seed)
        model.fit(X_fold_train[:, selected], y_fold_train)
        y_pred_val = model.predict(X_fold_val[:, selected])
        fold_losses.append(float(mean_absolute_error(y_fold_val, y_pred_val)))

    return float(np.mean(fold_losses))


def select_features_with_cv(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    model_type: str,
    seed: int,
    dataset_fingerprint: str,
    configuration_signature: dict[str, object],
) -> FeatureSelectionResult:
    enabled = bool(get_cfg("ablation.pipeline.feature_selection.enabled", True))
    if not enabled:
        all_idx = np.arange(X_train.shape[1], dtype=int)
        return FeatureSelectionResult(
            X_train=X_train,
            X_test=X_test,
            selected_feature_indices=all_idx,
            selected_feature_names=list(feature_names),
            removed_constant_features=[],
            removed_correlated_features=[],
            cv_mean_mae_by_k={},
            best_k=int(X_train.shape[1]),
            used_cache=False,
            cache_path="",
            enabled=False,
        )

    near_constant_var = float(
        get_cfg("ablation.pipeline.feature_selection.near_constant_variance_threshold", 1e-8)
    )
    corr_threshold = float(get_cfg("ablation.pipeline.feature_selection.high_correlation_threshold", 0.995))
    cv_splits_cfg = int(get_cfg("ablation.pipeline.feature_selection.cv_splits", 5))
    candidate_k_cfg = get_cfg("ablation.pipeline.feature_selection.candidate_feature_counts", [8, 16, 32, 64, 128])
    include_all = bool(get_cfg("ablation.pipeline.feature_selection.include_all_features_candidate", True))
    cache_enabled = bool(get_cfg("ablation.pipeline.feature_selection.cache.enabled", True))
    cache_dir = str(
        Path(get_cfg("ablation.artifacts.base_dir", "model/ablation/artifacts")) / "selected_features"
    )

    en_alpha = float(get_cfg("ablation.pipeline.feature_selection.elastic_net.alpha", 0.001))
    en_l1_ratio = float(get_cfg("ablation.pipeline.feature_selection.elastic_net.l1_ratio", 0.5))
    en_max_iter = int(get_cfg("ablation.pipeline.feature_selection.elastic_net.max_iter", 10000))

    (
        X_train_pruned,
        X_test_pruned,
        pruned_feature_names,
        removed_constant,
        removed_corr,
        kept_original_indices,
    ) = _redundancy_prune(
        X_train=X_train,
        X_test=X_test,
        feature_names=feature_names,
        near_constant_variance_threshold=near_constant_var,
        high_correlation_threshold=corr_threshold,
    )

    n_features = X_train_pruned.shape[1]
    if n_features == 0:
        raise RuntimeError("Feature pruning removed all features; adjust feature-selection thresholds.")

    candidate_k: list[int] = []
    for raw_k in candidate_k_cfg:
        try:
            k = int(raw_k)
        except Exception:
            continue
        if k <= 0:
            continue
        candidate_k.append(min(k, n_features))
    if include_all:
        candidate_k.append(n_features)
    candidate_k = sorted(set(candidate_k))
    if not candidate_k:
        candidate_k = [n_features]

    cache_key_payload = {
        "dataset_fingerprint": dataset_fingerprint,
        "model_type": model_type,
        "seed": int(seed),
        "configuration_signature": configuration_signature,
        "n_input_features": len(feature_names),
        "feature_names": feature_names,
        "near_constant_variance_threshold": near_constant_var,
        "high_correlation_threshold": corr_threshold,
        "cv_splits": cv_splits_cfg,
        "candidate_feature_counts": candidate_k,
        "elastic_net": {
            "alpha": en_alpha,
            "l1_ratio": en_l1_ratio,
            "max_iter": en_max_iter,
        },
    }
    cache_digest = hashlib.sha256(
        json.dumps(cache_key_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:24]
    cache_path = str(Path(cache_dir) / f"selected_features_{cache_digest}.json")

    if cache_enabled and Path(cache_path).exists():
        cached = json.loads(Path(cache_path).read_text(encoding="utf-8"))
        cached_names = cached.get("selected_feature_names", [])
        if isinstance(cached_names, list):
            name_to_pruned_idx = {name: i for i, name in enumerate(pruned_feature_names)}
            selected_pruned_idx = [name_to_pruned_idx[name] for name in cached_names if name in name_to_pruned_idx]
            if selected_pruned_idx:
                selected_pruned_idx_arr = np.array(sorted(set(selected_pruned_idx)), dtype=int)
                selected_original_idx = kept_original_indices[selected_pruned_idx_arr]
                return FeatureSelectionResult(
                    X_train=X_train[:, selected_original_idx],
                    X_test=X_test[:, selected_original_idx],
                    selected_feature_indices=selected_original_idx,
                    selected_feature_names=[feature_names[i] for i in selected_original_idx],
                    removed_constant_features=removed_constant,
                    removed_correlated_features=removed_corr,
                    cv_mean_mae_by_k={int(k): float(v) for k, v in cached.get("cv_mean_mae_by_k", {}).items()},
                    best_k=int(cached.get("best_k", len(selected_original_idx))),
                    used_cache=True,
                    cache_path=cache_path,
                    enabled=True,
                )

    n_samples = X_train_pruned.shape[0]
    max_valid_splits = n_samples - 1
    cv_splits = min(cv_splits_cfg, max_valid_splits)
    if cv_splits < 2:
        cv_splits = 0

    cv_scores: dict[int, float] = {}
    if cv_splits >= 2 and len(candidate_k) > 1:
        for k in candidate_k:
            cv_scores[k] = _cv_mean_mae_for_k(
                X_train=X_train_pruned,
                y_train=y_train,
                model_type=model_type,
                seed=seed,
                k=k,
                cv_splits=cv_splits,
                en_alpha=en_alpha,
                en_l1_ratio=en_l1_ratio,
                en_max_iter=en_max_iter,
            )
        best_k = min(sorted(cv_scores.keys()), key=lambda item: cv_scores[item])
    else:
        best_k = n_features

    feature_order, feature_scores = _rank_features_with_elastic_net(
        X=X_train_pruned,
        y=y_train,
        seed=seed,
        alpha=en_alpha,
        l1_ratio=en_l1_ratio,
        max_iter=en_max_iter,
    )
    selected_pruned_idx = _choose_k_indices(feature_order, feature_scores, best_k)
    selected_pruned_idx = np.array(sorted(set(selected_pruned_idx.tolist())), dtype=int)
    selected_original_idx = kept_original_indices[selected_pruned_idx]
    selected_feature_names = [feature_names[i] for i in selected_original_idx]

    if cache_enabled:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        cache_payload = {
            "selected_feature_names": selected_feature_names,
            "best_k": int(best_k),
            "cv_mean_mae_by_k": {str(k): float(v) for k, v in cv_scores.items()},
            "removed_constant_features": removed_constant,
            "removed_correlated_features": removed_corr,
            "configuration_signature": configuration_signature,
            "dataset_fingerprint": dataset_fingerprint,
        }
        Path(cache_path).write_text(json.dumps(cache_payload, indent=2, sort_keys=True), encoding="utf-8")

    return FeatureSelectionResult(
        X_train=X_train[:, selected_original_idx],
        X_test=X_test[:, selected_original_idx],
        selected_feature_indices=selected_original_idx,
        selected_feature_names=selected_feature_names,
        removed_constant_features=removed_constant,
        removed_correlated_features=removed_corr,
        cv_mean_mae_by_k=cv_scores,
        best_k=int(best_k),
        used_cache=False,
        cache_path=cache_path,
        enabled=True,
    )