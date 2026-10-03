"""Devaluation probability model using documented priors and regularized logistic regression."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from parity.features import FEATURE_COLUMNS
from parity.loaders import load_priors
from parity.schemas import DataValidationError


def sigmoid(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Numerically stable logistic sigmoid function."""
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


class DevaluationModel:
    """Predicts devaluation probability over 12- and 24-month horizons.

    Parameters
    ----------
    horizon : int, default 12
        Forecast horizon in months (typically 12 or 24).
    priors_path : Optional[Union[str, Path]], default None
        Path to priors.yaml. If None, default config is loaded.
    """

    def __init__(
        self,
        horizon: int = 12,
        priors_path: Optional[Union[str, Path]] = None,
    ):
        if horizon not in (12, 24):
            raise ValueError(f"horizon must be 12 or 24; got {horizon}")
        self.horizon = horizon
        self.priors_path = priors_path
        self.feature_names: List[str] = list(FEATURE_COLUMNS)

        # Load priors from config
        self._load_priors()

        # State flags
        self.is_fitted: bool = False
        self.fitted_intercept: Optional[float] = None
        self.fitted_coefficients: Optional[Dict[str, float]] = None
        self.cv_metrics_: Optional[Dict[str, Any]] = None
        self.sklearn_model_: Optional[LogisticRegression] = None

    def _load_priors(self) -> None:
        """Load and set prior coefficients from priors.yaml."""
        priors_dict = load_priors(self.priors_path)
        deval_priors = priors_dict.get("devaluation_probability", {})

        intercept_key = f"intercept_{self.horizon}m"
        self.prior_intercept: float = float(deval_priors.get(intercept_key, -2.0 if self.horizon == 12 else -1.2))

        self.prior_coefficients: Dict[str, float] = {
            "parallel_premium": float(deval_priors.get("coef_parallel_premium", 3.5)),
            "reserves_cover": float(deval_priors.get("coef_reserves_cover", -0.4)),
            "reserves_change_3m": float(deval_priors.get("coef_reserves_change_3m", -1.5)),
            "ca_pct_gdp": float(deval_priors.get("coef_ca_pct_gdp", -0.08)),
            "rer_misalignment": float(deval_priors.get("coef_rer_misalignment", 1.2)),
            "rate_differential": float(deval_priors.get("coef_rate_differential", -0.05)),
            "imf_program": float(deval_priors.get("coef_imf_program", -0.6)),
            "imf_review_off_track": float(deval_priors.get("coef_imf_review_off_track", 1.0)),
        }

    @property
    def coefficients(self) -> Dict[str, float]:
        """Return active coefficients (fitted if fitted, otherwise documented priors)."""
        if self.is_fitted and self.fitted_coefficients is not None:
            return dict(self.fitted_coefficients)
        return dict(self.prior_coefficients)

    @property
    def coef_(self) -> Dict[str, float]:
        """Convenience alias for coefficients matching scikit-learn convention."""
        return self.coefficients

    @property
    def intercept(self) -> float:
        """Return active intercept (fitted if fitted, otherwise documented prior)."""
        if self.is_fitted and self.fitted_intercept is not None:
            return self.fitted_intercept
        return self.prior_intercept

    @property
    def intercept_(self) -> float:
        """Convenience alias for intercept matching scikit-learn convention."""
        return self.intercept

    @property
    def priors_(self) -> Dict[str, float]:
        """Return dictionary of prior coefficients and intercept."""
        res = {f"coef_{k}": v for k, v in self.prior_coefficients.items()}
        res[f"intercept_{self.horizon}m"] = self.prior_intercept
        return res

    def _prepare_feature_array(self, X: Union[pd.DataFrame, Dict[str, Any]]) -> np.ndarray:
        """Extract and align feature values into a 2D numpy array."""
        if isinstance(X, dict):
            row = [float(X.get(col, 0.0)) for col in self.feature_names]
            return np.array([row], dtype=float)

        if isinstance(X, pd.DataFrame):
            missing_cols = [c for c in self.feature_names if c not in X.columns]
            if missing_cols:
                raise DataValidationError(
                    f"Input DataFrame is missing required feature columns: {missing_cols}. "
                    f"Required: {self.feature_names}"
                )
            return X[self.feature_names].to_numpy(dtype=float)

        raise TypeError(f"Expected DataFrame or dict, got {type(X).__name__}")

    def predict_proba(self, X: Union[pd.DataFrame, Dict[str, Any]]) -> Union[float, np.ndarray]:
        """Predict devaluation probability over the model horizon.

        Parameters
        ----------
        X : Union[pd.DataFrame, Dict[str, Any]]
            Features DataFrame or single observation dictionary.

        Returns
        -------
        Union[float, np.ndarray]
            Probability of devaluation (scalar float if single dict, else 1D array).
        """
        is_single_dict = isinstance(X, dict)
        X_mat = self._prepare_feature_array(X)

        coef_vec = np.array([self.coefficients[c] for c in self.feature_names], dtype=float)
        logits = self.intercept + np.dot(X_mat, coef_vec)
        probs = sigmoid(logits)

        if is_single_dict:
            return float(probs[0])
        return probs

    def fit(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        C: float = 1.0,
        cv_splits: int = 3,
        date_col: str = "date",
    ) -> "DevaluationModel":
        """Fit regularized logistic regression on lagged features using time-series CV.

        Parameters
        ----------
        df : pd.DataFrame
            Enriched panel DataFrame containing FEATURE_COLUMNS, target_col, and date_col.
        target_col : Optional[str], default None
            Target column name. If None, defaults to f'target_devaluation_{self.horizon}m'.
        C : float, default 1.0
            Inverse regularization strength (L2 / Ridge penalty).
        cv_splits : int, default 3
            Number of time-series cross-validation splits (no random splits).
        date_col : str, default 'date'
            Date column used to enforce strict chronological ordering.

        Returns
        -------
        DevaluationModel
            Self instance with fitted coefficients and cross-validation metrics.
        """
        if target_col is None:
            target_col = f"target_devaluation_{self.horizon}m"

        if target_col not in df.columns:
            raise DataValidationError(f"Target column '{target_col}' not found in training DataFrame")

        # Sort strictly by date to prevent lookahead leakage
        train_df = df.copy()
        if date_col in train_df.columns:
            train_df = train_df.sort_values(date_col).reset_index(drop=True)

        # Drop right-censored observations (where target is NaN)
        valid_mask = train_df[target_col].notna()
        clean_df = train_df[valid_mask].copy()

        if clean_df.empty:
            raise DataValidationError(f"No valid non-censored observations for target '{target_col}'")

        y = clean_df[target_col].to_numpy(dtype=int)
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            warnings.warn(
                f"Training target '{target_col}' contains only one class ({unique_classes[0]}). "
                "Retaining documented priors.",
                UserWarning,
                stacklevel=2,
            )
            return self

        X_mat = self._prepare_feature_array(clean_df)

        # Time-series cross validation (no random splits)
        cv_scores: List[float] = []
        cv_brier: List[float] = []

        if len(clean_df) > cv_splits * 10:
            tscv = TimeSeriesSplit(n_splits=cv_splits)
            for train_idx, val_idx in tscv.split(X_mat):
                X_tr, y_tr = X_mat[train_idx], y[train_idx]
                X_va, y_va = X_mat[val_idx], y[val_idx]

                if len(np.unique(y_tr)) > 1 and len(np.unique(y_va)) > 1:
                    fold_scaler = StandardScaler()
                    X_tr_s = fold_scaler.fit_transform(X_tr)
                    X_va_s = fold_scaler.transform(X_va)
                    fold_clf = LogisticRegression(C=C, solver="lbfgs", max_iter=1000)
                    fold_clf.fit(X_tr_s, y_tr)
                    preds = fold_clf.predict_proba(X_va_s)[:, 1]
                    cv_scores.append(float(roc_auc_score(y_va, preds)))
                    cv_brier.append(float(brier_score_loss(y_va, preds)))

        # Final fit on full dataset with standard scaler
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_mat)
        final_clf = LogisticRegression(C=C, solver="lbfgs", max_iter=2000)
        final_clf.fit(X_s, y)

        scale = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
        unscaled_coef = final_clf.coef_[0] / scale
        unscaled_intercept = final_clf.intercept_[0] - np.sum(final_clf.coef_[0] * scaler.mean_ / scale)

        self.sklearn_model_ = final_clf
        self.fitted_intercept = float(unscaled_intercept)
        self.fitted_coefficients = {
            col: float(coef) for col, coef in zip(self.feature_names, unscaled_coef)
        }
        self.is_fitted = True

        self.cv_metrics_ = {
            "mean_cv_auc": float(np.mean(cv_scores)) if cv_scores else None,
            "mean_cv_brier": float(np.mean(cv_brier)) if cv_brier else None,
            "n_samples": int(len(clean_df)),
            "event_rate": float(np.mean(y)),
        }

        return self

    def evaluate(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        n_bins: int = 5,
    ) -> Dict[str, Any]:
        """Compute calibration curve, Brier score, and AUC on a test or panel dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Panel DataFrame with features and target.
        target_col : Optional[str], default None
            Target column name. If None, defaults to f'target_devaluation_{self.horizon}m'.
        n_bins : int, default 5
            Number of bins for calibration curve.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing brier_score, roc_auc, prob_true, prob_pred, event_rate.
        """
        if target_col is None:
            target_col = f"target_devaluation_{self.horizon}m"

        if target_col not in df.columns:
            raise DataValidationError(f"Target column '{target_col}' not found")

        valid_mask = df[target_col].notna()
        eval_df = df[valid_mask]
        if eval_df.empty:
            raise DataValidationError("Evaluation DataFrame contains only NaN target values")

        y_true = eval_df[target_col].to_numpy(dtype=int)
        y_pred = self.predict_proba(eval_df)

        metrics: Dict[str, Any] = {
            "n_samples": int(len(eval_df)),
            "event_rate": float(np.mean(y_true)),
            "brier_score": float(brier_score_loss(y_true, y_pred)),
        }

        if len(np.unique(y_true)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred))
            prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy="uniform")
            metrics["calibration_prob_true"] = prob_true.tolist()
            metrics["calibration_prob_pred"] = prob_pred.tolist()
        else:
            metrics["roc_auc"] = None
            metrics["calibration_prob_true"] = []
            metrics["calibration_prob_pred"] = []

        return metrics
