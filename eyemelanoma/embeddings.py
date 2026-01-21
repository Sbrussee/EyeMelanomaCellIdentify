"""Embedding and clustering utilities for slide-level vectors."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from eyemelanoma.config import ClusteringConfig


def _best_kmeans(X: np.ndarray, k_min: int, k_max: int, random_state: int) -> tuple[int, KMeans, np.ndarray]:
    best_k = k_min
    best_score = -1.0
    best_model = None
    best_labels = None
    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        labels = model.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X, labels)
        if score > best_score:
            best_k, best_score, best_model, best_labels = k, score, model, labels
    if best_model is None:
        best_model = KMeans(n_clusters=k_min, n_init=20, random_state=random_state).fit(X)
        best_labels = best_model.labels_
    return best_k, best_model, best_labels


def _maybe_umap():
    try:
        from umap import UMAP as UMAPClass
        return UMAPClass
    except Exception:
        try:
            from umap.umap_ import UMAP as UMAPClass
            return UMAPClass
        except Exception:
            return None


def run_embeddings_and_clustering(
    X: np.ndarray,
    label: str,
    out_dir: Path,
    config: ClusteringConfig,
) -> Dict[str, Optional[np.ndarray]]:
    out_dir.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=min(50, Xs.shape[1]))
    Xp = pca.fit_transform(Xs)

    k, _, labels = _best_kmeans(Xp, config.k_min, config.k_max, config.random_state)

    umap_cls = _maybe_umap()
    X_umap = None
    if umap_cls is not None:
        umap = umap_cls(n_neighbors=15, min_dist=0.2, random_state=config.random_state)
        X_umap = umap.fit_transform(Xp)
        plt.figure(figsize=(6, 5))
        plt.scatter(X_umap[:, 0], X_umap[:, 1], c=labels, s=28)
        plt.title(f"UMAP – {label} (k={k})")
        plt.tight_layout()
        plt.savefig(out_dir / f"{label}_umap.png", dpi=160)
        plt.close()

    tsne = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="pca", random_state=config.random_state)
    X_tsne = tsne.fit_transform(Xp)
    plt.figure(figsize=(6, 5))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, s=28)
    plt.title(f"t-SNE – {label} (k={k})")
    plt.tight_layout()
    plt.savefig(out_dir / f"{label}_tsne.png", dpi=160)
    plt.close()

    np.save(out_dir / f"{label}_pca.npy", Xp)
    np.save(out_dir / f"{label}_tsne.npy", X_tsne)
    np.save(out_dir / f"{label}_kmeans_labels.npy", labels)
    if X_umap is not None:
        np.save(out_dir / f"{label}_umap.npy", X_umap)

    return {"labels": labels, "umap": X_umap, "tsne": X_tsne}
