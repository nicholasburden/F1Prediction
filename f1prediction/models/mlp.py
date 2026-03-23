import torch
import torch.nn as nn
from torch import Tensor

from .base import F1PredictionModel


class LinearRegression(F1PredictionModel):
    """Pure linear regression: Linear(input_dim, 1), no activation."""

    def __init__(self, continuous_dim: int, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.linear = nn.Linear(continuous_dim + self.embed_dim, 1)

    def forward(self, x: Tensor, cat_ids: Tensor) -> Tensor:
        emb = self.embed_categorical(cat_ids)
        return self.linear(torch.cat([x, emb], dim=-1)).squeeze(-1)

    @property
    def name(self) -> str:
        return "LinearRegression"


class SingleLayerMLP(F1PredictionModel):
    """Linear(input_dim, 1) -> Sigmoid. Kept for backwards compatibility."""

    def __init__(self, continuous_dim: int, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.linear = nn.Linear(continuous_dim + self.embed_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor, cat_ids: Tensor) -> Tensor:
        emb = self.embed_categorical(cat_ids)
        return self.sigmoid(self.linear(torch.cat([x, emb], dim=-1))).squeeze(-1)

    @property
    def name(self) -> str:
        return "SingleLayerMLP"


class MLP(F1PredictionModel):
    """Multi-layer perceptron with ReLU, BatchNorm, and Dropout."""

    def __init__(
        self,
        continuous_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if hidden_dims is None:
            hidden_dims = [128, 64]

        input_dim = continuous_dim + self.embed_dim
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
        self._hidden_dims = hidden_dims

    def forward(self, x: Tensor, cat_ids: Tensor) -> Tensor:
        emb = self.embed_categorical(cat_ids)
        return self.net(torch.cat([x, emb], dim=-1)).squeeze(-1)

    @property
    def name(self) -> str:
        dims = "x".join(str(d) for d in self._hidden_dims)
        return f"MLP({dims})"
