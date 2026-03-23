from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class F1PredictionModel(ABC, nn.Module):
    """Base model with learned embeddings for driver and team identity.

    Index 0 is reserved for unknown (unseen) entities, so vocab_size
    should be ``len(vocab) + 1``.
    """

    def __init__(
        self,
        driver_vocab_size: int = 1,
        team_vocab_size: int = 1,
        driver_embed_dim: int = 8,
        team_embed_dim: int = 4,
        normalize_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self.driver_embedding = nn.Embedding(driver_vocab_size, driver_embed_dim)
        self.team_embedding = nn.Embedding(team_vocab_size, team_embed_dim)
        self.embed_dim = driver_embed_dim + team_embed_dim
        self.normalize_embeddings = normalize_embeddings

    def embed_categorical(self, cat_ids: Tensor) -> Tensor:
        """(B, 2) -> (B, embed_dim) — concatenate driver + team embeddings."""
        driver_emb = self.driver_embedding(cat_ids[:, 0])
        team_emb = self.team_embedding(cat_ids[:, 1])
        if self.normalize_embeddings:
            driver_emb = F.normalize(driver_emb, dim=-1)
            team_emb = F.normalize(team_emb, dim=-1)
        return torch.cat([driver_emb, team_emb], dim=-1)

    @abstractmethod
    def forward(self, x: Tensor, cat_ids: Tensor) -> Tensor:
        """(B, D_continuous), (B, 2) -> (B,)"""
        ...

    @property
    @abstractmethod
    def name(self) -> str: ...
