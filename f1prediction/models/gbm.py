"""Gradient boosting model wrapper for XGBoost and LightGBM."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np

from ..data.extraction import ExtractedData

logger = logging.getLogger(__name__)


class GBMModel:
    """Gradient boosting model for F1 position prediction.

    Wraps XGBoost or LightGBM behind a unified interface. Driver and team
    identity are passed as integer-encoded categorical columns appended to
    the continuous feature vector.
    """

    def __init__(
        self,
        backend: str = "xgboost",
        *,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 5,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        early_stopping_rounds: int = 50,
        n_continuous: int = 0,
        seed: int = 42,
    ) -> None:
        self.backend = backend
        self.n_continuous = n_continuous
        self.early_stopping_rounds = early_stopping_rounds
        self._model = None

        if backend == "xgboost":
            try:
                import xgboost as xgb  # noqa: F401
            except ImportError:
                raise ImportError(
                    "XGBoost is not installed. Install with: uv add xgboost"
                )
            from xgboost import XGBRegressor

            self._model = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                min_child_weight=min_child_weight,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                objective="reg:squarederror",
                random_state=seed,
                enable_categorical=True,
            )
        elif backend == "lightgbm":
            try:
                import lightgbm as lgb  # noqa: F401
            except ImportError:
                raise ImportError(
                    "LightGBM is not installed. Install with: uv add lightgbm"
                )
            from lightgbm import LGBMRegressor

            self._model = LGBMRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                min_child_weight=min_child_weight,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                objective="regression",
                random_state=seed,
                verbosity=-1,
            )
        else:
            raise ValueError(f"Unknown GBM backend: {backend}")

    @property
    def name(self) -> str:
        return "XGBoost" if self.backend == "xgboost" else "LightGBM"

    def _build_dataframe(
        self, data: ExtractedData, feature_names: list[str],
    ):
        """Build a pandas DataFrame with proper types for XGBoost."""
        import pandas as pd

        df = pd.DataFrame(data.features, columns=feature_names)
        df["driver_id"] = pd.Categorical(data.cat_ids[:, 0])
        df["team_id"] = pd.Categorical(data.cat_ids[:, 1])
        return df

    def _build_dataframe_raw(
        self, features: np.ndarray, cat_ids: np.ndarray, feature_names: list[str],
    ):
        """Build a pandas DataFrame from raw arrays."""
        import pandas as pd

        df = pd.DataFrame(features, columns=feature_names)
        df["driver_id"] = pd.Categorical(cat_ids[:, 0])
        df["team_id"] = pd.Categorical(cat_ids[:, 1])
        return df

    def _build_numpy_matrix(self, data: ExtractedData) -> np.ndarray:
        """Concatenate continuous features and categorical IDs for LightGBM."""
        return np.concatenate(
            [data.features, data.cat_ids.astype(np.float32)], axis=1,
        )

    def fit(
        self,
        train_data: ExtractedData,
        val_data: ExtractedData | None = None,
        feature_names: list[str] | None = None,
    ) -> dict[str, float]:
        """Train the model. Returns validation metrics if val_data provided."""
        y_train = train_data.targets
        cont_names = list(feature_names or [])
        all_names = cont_names + ["driver_id", "team_id"]

        fit_kwargs: dict = {}

        if self.backend == "xgboost":
            X_train_df = self._build_dataframe(train_data, cont_names)

            if val_data is not None:
                X_val_df = self._build_dataframe(val_data, cont_names)
                fit_kwargs["eval_set"] = [(X_val_df, val_data.targets)]

            self._model.set_params(
                early_stopping_rounds=self.early_stopping_rounds,
            )
            self._model.fit(X_train_df, y_train, verbose=50, **fit_kwargs)
            self._feature_names = cont_names

        elif self.backend == "lightgbm":
            X_train = self._build_numpy_matrix(train_data)
            cat_indices = [len(all_names) - 2, len(all_names) - 1]

            if val_data is not None:
                X_val = self._build_numpy_matrix(val_data)
                fit_kwargs["eval_set"] = [(X_val, val_data.targets)]
                fit_kwargs["callbacks"] = [
                    _lgbm_early_stopping(self.early_stopping_rounds),
                    _lgbm_log_evaluation(50),
                ]

            self._model.fit(
                X_train,
                y_train,
                feature_name=all_names,
                categorical_feature=cat_indices,
                **fit_kwargs,
            )
            self._feature_names = cont_names

        return {}

    def predict(self, features: np.ndarray, cat_ids: np.ndarray) -> np.ndarray:
        """Predict from numpy arrays. Returns predictions clipped to [0, 1]."""
        cont_names = getattr(self, "_feature_names", [f"f{i}" for i in range(features.shape[1])])
        if self.backend == "xgboost":
            df = self._build_dataframe_raw(features, cat_ids, cont_names)
            preds = self._model.predict(df)
        else:
            import pandas as pd

            all_names = cont_names + ["driver_id", "team_id"]
            X = np.concatenate([features, cat_ids.astype(np.float32)], axis=1)
            df = pd.DataFrame(X, columns=all_names)
            preds = self._model.predict(df)

        return np.clip(preds, 0.0, 1.0)

    def feature_importance(self, feature_names: list[str]) -> dict[str, float]:
        """Return feature importance scores."""
        all_names = list(feature_names) + ["driver_id", "team_id"]
        importances = self._model.feature_importances_
        return dict(zip(all_names, importances.tolist()))

    def save(self, path: Path) -> None:
        """Save model to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "backend": self.backend,
                "model": self._model,
                "n_continuous": self.n_continuous,
                "feature_names": getattr(self, "_feature_names", []),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> GBMModel:
        """Load a saved model."""
        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj.backend = data["backend"]
        obj._model = data["model"]
        obj.n_continuous = data.get("n_continuous", 0)
        obj._feature_names = data.get("feature_names", [])
        obj.early_stopping_rounds = 50
        return obj


def _lgbm_early_stopping(stopping_rounds: int):
    """Get LightGBM early stopping callback."""
    from lightgbm import early_stopping
    return early_stopping(stopping_rounds)


def _lgbm_log_evaluation(period: int):
    """Get LightGBM logging callback."""
    from lightgbm import log_evaluation
    return log_evaluation(period)
