from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from f1prediction.training.metrics import MetricConfig


class MLPModelConfig(BaseModel):
    type: Literal["mlp"]
    embedding_dim: int
    hidden_dim: int
    num_hidden_layers: int


ModelConfig = MLPModelConfig


class TrainingConfig(BaseModel):
    seed: int
    lr: float
    num_epochs: int | None = None
    patience: int
    min_delta: float
    batch_size: int
    gradient_accumulation: int
    driver_dropout: float
    loss: Literal["mae", "mse"]
    optimizer: Literal["adam"]
    device: Literal["cpu", "mps"]
    target_sessions: list[str]
    years: list[int]
    train_metrics: list[MetricConfig] = [MetricConfig(type="mae")] + [
        MetricConfig(type="within_k", k=float(k)) for k in range(1, 7)
    ]
    val_metrics: list[MetricConfig] = [MetricConfig(type="mae")] + [
        MetricConfig(type="within_k", k=float(k)) for k in range(1, 7)
    ]
    wandb_group: str | None = None
    data_dir: Path
    output_dir: Path


_SHORT_NAMES: dict[str, str] = {
    "gradient_accumulation": "ga",
    "driver_dropout": "ddrop",
    "min_delta": "mdelta",
    "num_epochs": "ep",
    "batch_size": "bs",
    "embedding_dim": "edim",
    "hidden_dim": "hdim",
    "num_hidden_layers": "nlayers",
    "optimizer": "opt",
    "patience": "pat",
}

_EXCLUDED_FIELDS = {
    "train_metrics",
    "val_metrics",
    "target_sessions",
    "years",
    "data_dir",
    "output_dir",
    "wandb_group",
}


def config_name(config: "Config") -> str:
    parts: list[str] = []
    for sub in [config.training, config.model]:
        for k, v in sub.model_dump(exclude=_EXCLUDED_FIELDS).items():
            short = _SHORT_NAMES.get(k, k)
            parts.append(f"{short}_{v}")
    return "__".join(parts)


class Config(BaseModel):
    training: TrainingConfig
    model: MLPModelConfig

    def run_name(self) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"{config_name(self)}_{ts}"

    def run_dir(self) -> Path:
        return self.training.output_dir / self.run_name()

    def to_flat_dict(self) -> dict[str, object]:
        d: dict[str, object] = {}
        for prefix, sub in [("training", self.training), ("model", self.model)]:
            for k, v in sub.model_dump().items():
                d[f"{prefix}/{k}"] = v
        return d
