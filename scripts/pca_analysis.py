#!/usr/bin/env python
"""PCA analysis of the F1 prediction model's feature space.

Loads training data (2020-2024, core+form tiers, lookback=3), extracts
normalized features, runs PCA, and produces visualizations:
  1. Scree plot (explained variance per component)
  2. Biplot of PC1 vs PC2, colored by finishing position
  3. Feature loadings heatmap for top PCs
  4. Prints top features by loading magnitude for PC1, PC2, PC3

Usage:
    uv run python scripts/pca_analysis.py
"""

import logging
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from f1prediction.data.dataset import F1RaceDataset
from f1prediction.data.history import build_history_table
from f1prediction.data.normalization import compute_stats
from f1prediction.data.registry import REGISTRY
from f1prediction.training.splits import (
    build_vocabularies,
    generate_samples,
    get_event_order,
    split_events,
)

# Import features to trigger registration
import f1prediction.data.features  # noqa: F401

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────
DATA_DIR = Path("data")
YEARS = list(range(2020, 2025))
LOOKBACK = 3
SEED = 42
TRAIN_FRAC = 0.7
VAL_FRAC = 0.15
MAX_DRIVERS = 22
OUTPUT_DIR = Path("runs/pca")
N_TOP_PCS = 8  # number of PCs for the heatmap
N_TOP_FEATURES = 10  # features to print per PC


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Enable only core + form tiers
    REGISTRY.enable_categories(["core", "form"])
    feature_names = REGISTRY.feature_names
    logger.info("Enabled %d features (core+form): %s", len(feature_names), feature_names[:5])

    # ── Discover events & build history ──────────────────────────────────
    all_events = get_event_order(DATA_DIR, YEARS)
    logger.info("Found %d events across %s", len(all_events), YEARS)

    logger.info("Building history tables (lookback=%d)...", LOOKBACK)
    history_table, team_history_table = build_history_table(DATA_DIR, all_events, lookback=LOOKBACK)
    logger.info(
        "History: %d driver entries, %d team entries",
        len(history_table), len(team_history_table),
    )

    # ── Split events & generate training samples ─────────────────────────
    train_events, val_events, test_events = split_events(
        all_events, TRAIN_FRAC, VAL_FRAC, seed=SEED,
    )
    logger.info(
        "Split: %d train, %d val, %d test events",
        len(train_events), len(val_events), len(test_events),
    )

    train_samples = generate_samples(DATA_DIR, train_events)
    logger.info("Generated %d training samples", len(train_samples))

    # ── Build vocabularies ───────────────────────────────────────────────
    driver_vocab, team_vocab = build_vocabularies(DATA_DIR, YEARS)
    logger.info("Vocabularies: %d drivers, %d teams", len(driver_vocab), len(team_vocab))

    # ── Build unnormalized dataset to compute norm stats ─────────────────
    logger.info("Building unnormalized dataset for normalization stats...")
    norm_subset = train_samples
    if len(train_samples) > 500:
        norm_subset = random.sample(train_samples, 500)

    raw_ds = F1RaceDataset(
        DATA_DIR,
        norm_subset,
        REGISTRY,
        max_drivers=MAX_DRIVERS,
        driver_vocab=driver_vocab,
        team_vocab=team_vocab,
        history_table=history_table,
        team_history_table=team_history_table,
    )
    norm_stats = compute_stats(raw_ds.get_all_features(), REGISTRY.feature_names)
    logger.info("Computed normalization stats (%d features)", len(norm_stats.feature_names))

    # ── Build normalized training dataset ────────────────────────────────
    logger.info("Building normalized training dataset (%d samples)...", len(train_samples))
    train_ds = F1RaceDataset(
        DATA_DIR,
        train_samples,
        REGISTRY,
        norm_stats=norm_stats,
        max_drivers=MAX_DRIVERS,
        driver_vocab=driver_vocab,
        team_vocab=team_vocab,
        history_table=history_table,
        team_history_table=team_history_table,
    )

    features_tensor = train_ds.get_all_features()  # (N, D)
    targets_tensor = train_ds._targets               # (N,)

    X = features_tensor.numpy()
    positions = targets_tensor.numpy() * MAX_DRIVERS  # denormalize to actual positions
    N, D = X.shape
    logger.info("Feature matrix: %d samples x %d features", N, D)

    # ── Run PCA ──────────────────────────────────────────────────────────
    n_components = min(N_TOP_PCS, D)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    logger.info(
        "PCA: %d components explain %.1f%% of total variance",
        n_components, pca.explained_variance_ratio_.sum() * 100,
    )

    # Also fit a full PCA for the scree plot
    pca_full = PCA().fit(X)

    # ── Create output directory ──────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Scree plot ────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    explained = pca_full.explained_variance_ratio_ * 100
    cumulative = np.cumsum(explained)
    n_show = min(20, len(explained))

    ax1.bar(range(1, n_show + 1), explained[:n_show], color="steelblue", alpha=0.8)
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance (%)")
    ax1.set_title("Scree Plot: Variance per Component")
    ax1.set_xticks(range(1, n_show + 1))

    ax2.plot(range(1, n_show + 1), cumulative[:n_show], "o-", color="darkorange", linewidth=2)
    ax2.axhline(y=90, color="gray", linestyle="--", alpha=0.5, label="90% threshold")
    ax2.axhline(y=95, color="gray", linestyle=":", alpha=0.5, label="95% threshold")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Explained Variance (%)")
    ax2.set_title("Cumulative Variance Explained")
    ax2.set_xticks(range(1, n_show + 1))
    ax2.legend()

    fig.tight_layout()
    scree_path = OUTPUT_DIR / "scree_plot.png"
    fig.savefig(scree_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved scree plot to %s", scree_path)

    # ── 2. Biplot: PC1 vs PC2 colored by finishing position ──────────────
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=positions,
        cmap="RdYlGn_r",
        alpha=0.4,
        s=8,
        edgecolors="none",
    )
    cbar = fig.colorbar(scatter, ax=ax, label="Finishing Position")

    # Overlay loading arrows for the top contributing features
    loadings = pca.components_[:2, :]  # (2, D)
    loading_magnitudes = np.sqrt(loadings[0] ** 2 + loadings[1] ** 2)
    top_loading_indices = np.argsort(loading_magnitudes)[-8:]  # top 8 features

    # Scale arrows to fit the plot
    pc1_range = X_pca[:, 0].max() - X_pca[:, 0].min()
    pc2_range = X_pca[:, 1].max() - X_pca[:, 1].min()
    scale = min(pc1_range, pc2_range) * 0.4 / max(loading_magnitudes[top_loading_indices])

    for idx in top_loading_indices:
        ax.annotate(
            feature_names[idx],
            xy=(loadings[0, idx] * scale, loadings[1, idx] * scale),
            xytext=(loadings[0, idx] * scale * 1.15, loadings[1, idx] * scale * 1.15),
            color="black",
            fontsize=7,
            ha="center",
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
        )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    ax.set_title("PCA Biplot: PC1 vs PC2 (colored by finishing position)")
    ax.axhline(y=0, color="gray", linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color="gray", linewidth=0.5, alpha=0.5)

    fig.tight_layout()
    biplot_path = OUTPUT_DIR / "biplot_pc1_pc2.png"
    fig.savefig(biplot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved biplot to %s", biplot_path)

    # ── 3. Feature loadings heatmap ──────────────────────────────────────
    loadings_matrix = pca.components_  # (n_components, D)

    # Select features with highest absolute loading on any of the top PCs
    max_abs_loading = np.max(np.abs(loadings_matrix), axis=0)
    n_features_to_show = min(30, D)
    top_feature_indices = np.argsort(max_abs_loading)[-n_features_to_show:]
    # Sort them by their max loading for visual clarity
    top_feature_indices = top_feature_indices[np.argsort(max_abs_loading[top_feature_indices])]

    selected_names = [feature_names[i] for i in top_feature_indices]
    selected_loadings = loadings_matrix[:, top_feature_indices]  # (n_components, n_selected)

    fig, ax = plt.subplots(figsize=(14, max(8, n_features_to_show * 0.35)))
    im = ax.imshow(
        selected_loadings.T,
        cmap="RdBu_r",
        aspect="auto",
        vmin=-np.max(np.abs(selected_loadings)),
        vmax=np.max(np.abs(selected_loadings)),
    )
    ax.set_xticks(range(n_components))
    ax.set_xticklabels([f"PC{i+1}" for i in range(n_components)])
    ax.set_yticks(range(len(selected_names)))
    ax.set_yticklabels(selected_names, fontsize=8)
    ax.set_xlabel("Principal Component")
    ax.set_title(f"Feature Loadings Heatmap (top {n_features_to_show} features by max |loading|)")
    fig.colorbar(im, ax=ax, label="Loading", shrink=0.8)

    fig.tight_layout()
    heatmap_path = OUTPUT_DIR / "loadings_heatmap.png"
    fig.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved loadings heatmap to %s", heatmap_path)

    # ── 4. Print top features per PC ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("TOP FEATURES BY LOADING MAGNITUDE")
    print("=" * 70)

    for pc_idx in range(min(3, n_components)):
        pc_loadings = loadings_matrix[pc_idx]
        sorted_indices = np.argsort(np.abs(pc_loadings))[::-1]

        print(f"\nPC{pc_idx + 1} ({pca.explained_variance_ratio_[pc_idx]*100:.1f}% variance):")
        print(f"  {'Feature':<35s} {'Loading':>10s}")
        print(f"  {'-'*35} {'-'*10}")
        for rank, idx in enumerate(sorted_indices[:N_TOP_FEATURES]):
            print(f"  {feature_names[idx]:<35s} {pc_loadings[idx]:>10.4f}")

    # ── Summary stats ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PCA SUMMARY")
    print("=" * 70)
    print(f"  Total features:        {D}")
    print(f"  Total samples:         {N}")
    n_90 = np.searchsorted(cumulative, 90.0) + 1
    n_95 = np.searchsorted(cumulative, 95.0) + 1
    print(f"  Components for 90%:    {n_90}")
    print(f"  Components for 95%:    {n_95}")
    for i in range(min(5, n_components)):
        print(f"  PC{i+1} variance:          {pca.explained_variance_ratio_[i]*100:.1f}%")
    print(f"  Top {n_components} PCs total:     {pca.explained_variance_ratio_.sum()*100:.1f}%")

    print(f"\nPlots saved to {OUTPUT_DIR.resolve()}/")
    logger.info("PCA analysis complete.")


if __name__ == "__main__":
    main()
