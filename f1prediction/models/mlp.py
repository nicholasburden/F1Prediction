import torch
from torch import nn

from f1prediction.config import MLPModelConfig


class MLPModel(nn.Module):
    def __init__(
        self, cfg: MLPModelConfig, num_numerical_features: int, vocab_lens: list[int]
    ):
        super().__init__()
        self.embedding_layers = nn.ModuleList(
            [nn.Embedding(v, cfg.embedding_dim) for v in vocab_lens]
        )

        layers: list[nn.Module] = []
        in_dim = num_numerical_features + (cfg.embedding_dim * len(vocab_lens))
        for _ in range(cfg.num_hidden_layers):
            layers.append(nn.Linear(in_dim, cfg.hidden_dim))
            layers.append(nn.ReLU())
            in_dim = cfg.hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        embeddings: list[torch.Tensor] = [
            layer(col)
            for layer, col in zip(self.embedding_layers, cat_ids.unbind(dim=1))
        ]
        out: torch.Tensor = self.mlp(torch.concat([X] + embeddings, dim=-1))
        return out.squeeze(-1)
