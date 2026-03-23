from __future__ import annotations

import numpy as np
from torch import Tensor


class PositionMetrics:
    @staticmethod
    def compute(preds: Tensor, targets: Tensor, max_drivers: int = 20) -> dict[str, float]:
        """Compute position prediction metrics.

        Args:
            preds: normalized predictions in [0, 1], shape (N,)
            targets: normalized targets in [0, 1], shape (N,)
            max_drivers: denormalization factor
        """
        pred_pos = preds * max_drivers
        true_pos = targets * max_drivers
        pred_rounded = pred_pos.round()

        diff = (pred_pos - true_pos).abs()
        mae = diff.mean().item()
        rmse = diff.pow(2).mean().sqrt().item()

        # Top-3 accuracy: check if rounded prediction <= 3 when true position <= 3
        top3_correct = ((pred_rounded <= 3) & (true_pos <= 3)).float().sum()
        top3_total = (true_pos <= 3).float().sum()
        top3_acc = (top3_correct / top3_total).item() if top3_total > 0 else 0.0

        exact = (diff < 0.5).float().mean().item()

        return {
            "mae_positions": mae,
            "rmse_positions": rmse,
            "top3_accuracy": top3_acc,
            "exact_accuracy": exact,
        }


def compute_metrics_numpy(
    preds: np.ndarray, targets: np.ndarray, max_drivers: int = 20,
) -> dict[str, float]:
    """Compute position prediction metrics using numpy arrays."""
    pred_pos = preds * max_drivers
    true_pos = targets * max_drivers
    pred_rounded = np.round(pred_pos)

    diff = np.abs(pred_pos - true_pos)
    mae = float(diff.mean())
    rmse = float(np.sqrt((diff**2).mean()))

    top3_correct = ((pred_rounded <= 3) & (true_pos <= 3)).sum()
    top3_total = (true_pos <= 3).sum()
    top3_acc = float(top3_correct / top3_total) if top3_total > 0 else 0.0

    exact = float((diff < 0.5).mean())

    return {
        "mae_positions": mae,
        "rmse_positions": rmse,
        "top3_accuracy": top3_acc,
        "exact_accuracy": exact,
    }
