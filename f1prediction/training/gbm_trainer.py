"""Trainer for gradient boosting models."""

from __future__ import annotations

import logging
from pathlib import Path

from ..data.extraction import ExtractedData
from ..models.gbm import GBMModel
from .metrics import compute_metrics_numpy

logger = logging.getLogger(__name__)


class GBMTrainer:
    """Trains and evaluates a GBMModel."""

    def __init__(
        self,
        model: GBMModel,
        train_data: ExtractedData,
        val_data: ExtractedData,
        output_dir: Path = Path("runs"),
        max_drivers: int = 20,
        feature_names: list[str] | None = None,
    ) -> None:
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.output_dir = output_dir
        self.max_drivers = max_drivers
        self.feature_names = feature_names or []

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fit(self) -> dict[str, float]:
        """Train with early stopping. Returns validation metrics."""
        logger.info(
            "Training %s (train=%d, val=%d)...",
            self.model.name,
            len(self.train_data.targets),
            len(self.val_data.targets),
        )

        self.model.fit(
            self.train_data,
            val_data=self.val_data,
            feature_names=self.feature_names,
        )

        # Save model
        model_path = self.output_dir / "best_model.joblib"
        self.model.save(model_path)
        logger.info("Saved model to %s", model_path)

        # Log feature importance (top 15)
        importance = self.model.feature_importance(self.feature_names)
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        logger.info("Top features by importance:")
        for name, score in sorted_imp[:15]:
            logger.info("  %-30s %.4f", name, score)

        return self.evaluate(self.val_data)

    def evaluate(self, data: ExtractedData) -> dict[str, float]:
        """Evaluate model on a dataset."""
        preds = self.model.predict(data.features, data.cat_ids)
        return compute_metrics_numpy(preds, data.targets, self.max_drivers)
