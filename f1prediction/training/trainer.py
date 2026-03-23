from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

try:
    import wandb as _wandb
except ImportError:
    _wandb = None

from ..models.base import F1PredictionModel
from .metrics import PositionMetrics

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: F1PredictionModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 100,
        patience: int = 15,
        output_dir: Path = Path("runs"),
        max_drivers: int = 20,
        embed_l2: float = 0.0,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.patience = patience
        self.output_dir = output_dir
        self.max_drivers = max_drivers
        self.embed_l2 = embed_l2

        self.criterion = nn.MSELoss()
        self.optimizer = Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self) -> tuple[float, dict[str, float]]:
        """Run one training epoch. Returns (avg_loss, metrics)."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        all_preds = []
        all_targets = []
        for features, cat_ids, targets in self.train_loader:
            self.optimizer.zero_grad()
            preds = self.model(features, cat_ids)
            loss = self.criterion(preds, targets)
            if self.embed_l2 > 0:
                for emb in (self.model.driver_embedding, self.model.team_embedding):
                    loss = loss + self.embed_l2 * emb.weight.pow(2).mean()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            all_preds.append(preds.detach())
            all_targets.append(targets)
        avg_loss = total_loss / max(n_batches, 1)
        metrics = PositionMetrics.compute(
            torch.cat(all_preds), torch.cat(all_targets), self.max_drivers,
        )
        metrics["loss"] = avg_loss
        return avg_loss, metrics

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        all_preds = []
        all_targets = []
        total_loss = 0.0
        n_batches = 0

        for features, cat_ids, targets in loader:
            preds = self.model(features, cat_ids)
            total_loss += self.criterion(preds, targets).item()
            n_batches += 1
            all_preds.append(preds)
            all_targets.append(targets)

        preds_t = torch.cat(all_preds)
        targets_t = torch.cat(all_targets)
        metrics = PositionMetrics.compute(preds_t, targets_t, self.max_drivers)
        metrics["loss"] = total_loss / max(n_batches, 1)
        return metrics

    def fit(self) -> dict[str, float]:
        """Train with early stopping. Returns best validation metrics."""
        best_val_loss = float("inf")
        patience_counter = 0
        best_metrics: dict[str, float] = {}
        checkpoint_path = self.output_dir / "best_model.pt"

        for epoch in range(1, self.epochs + 1):
            train_loss, train_metrics = self.train_epoch()
            val_metrics = self.evaluate(self.val_loader)
            val_loss = val_metrics["loss"]

            logger.info(
                "Epoch %3d | train_loss=%.5f | train_mae=%.2f | "
                "val_loss=%.5f | val_mae=%.2f | val_top3_acc=%.2f",
                epoch,
                train_loss,
                train_metrics["mae_positions"],
                val_loss,
                val_metrics["mae_positions"],
                val_metrics["top3_accuracy"],
            )

            if _wandb and _wandb.run is not None:
                _wandb.log({
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/mae": train_metrics["mae_positions"],
                    "train/rmse": train_metrics["rmse_positions"],
                    "val/loss": val_loss,
                    "val/mae": val_metrics["mae_positions"],
                    "val/rmse": val_metrics["rmse_positions"],
                    "val/top3_accuracy": val_metrics["top3_accuracy"],
                    "val/exact_accuracy": val_metrics["exact_accuracy"],
                })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_metrics = val_metrics
                torch.save(self.model.state_dict(), checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(
                        "Early stopping at epoch %d (patience=%d)",
                        epoch,
                        self.patience,
                    )
                    break

        # Restore best model and re-evaluate val for unbiased metrics
        if checkpoint_path.exists():
            self.model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        return self.evaluate(self.val_loader)
