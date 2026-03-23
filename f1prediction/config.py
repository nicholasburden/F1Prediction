from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    data_dir: Path = Path("data")
    years: list[int] = field(default_factory=lambda: [2024])
    train_frac: float = 0.7
    val_frac: float = 0.15
    max_drivers: int = 22
    batch_size: int = 32
    num_workers: int = 0
    enabled_features: list[str] | None = None  # None = all
    unk_dropout: float = 0.15


@dataclass
class ModelConfig:
    # Backend selection: "torch", "xgboost", or "lightgbm"
    backend: str = "torch"

    # Neural net params (backend="torch")
    model_type: str = "mlp"
    hidden_dims: list[int] = field(default_factory=lambda: [128, 64])
    dropout: float = 0.1
    driver_embed_dim: int = 8
    team_embed_dim: int = 4
    normalize_embeddings: bool = False

    # Tree model params (backend="xgboost" or "lightgbm")
    n_estimators: int = 500
    max_depth: int = 6
    gbm_learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 5
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 50


@dataclass
class TrainingConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    embed_l2: float = 0.0
    epochs: int = 100
    patience: int = 15
    output_dir: Path = Path("runs")
    seed: int = 42


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
