from __future__ import annotations

from ..config import ModelConfig
from .base import F1PredictionModel
from .mlp import MLP, LinearRegression, SingleLayerMLP


def build_model(
    continuous_dim: int,
    cfg: ModelConfig,
    driver_vocab_size: int = 1,
    team_vocab_size: int = 1,
    seed: int = 42,
) -> F1PredictionModel:
    """Create a model from config.

    For torch backends returns an F1PredictionModel (nn.Module).
    For tree backends returns a GBMModel.
    """
    if cfg.backend in ("xgboost", "lightgbm"):
        from .gbm import GBMModel

        return GBMModel(
            backend=cfg.backend,
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.gbm_learning_rate,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            min_child_weight=cfg.min_child_weight,
            reg_alpha=cfg.reg_alpha,
            reg_lambda=cfg.reg_lambda,
            early_stopping_rounds=cfg.early_stopping_rounds,
            n_continuous=continuous_dim,
            seed=seed,
        )

    embed_kwargs = dict(
        driver_vocab_size=driver_vocab_size,
        team_vocab_size=team_vocab_size,
        driver_embed_dim=cfg.driver_embed_dim,
        team_embed_dim=cfg.team_embed_dim,
        normalize_embeddings=cfg.normalize_embeddings,
    )
    if cfg.model_type == "linear":
        return LinearRegression(continuous_dim, **embed_kwargs)
    if cfg.model_type == "single_layer_mlp":
        return SingleLayerMLP(continuous_dim, **embed_kwargs)
    if cfg.model_type == "mlp":
        return MLP(
            continuous_dim,
            hidden_dims=cfg.hidden_dims,
            dropout=cfg.dropout,
            **embed_kwargs,
        )
    raise ValueError(f"Unknown model_type: {cfg.model_type}")
