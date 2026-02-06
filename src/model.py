"""Discrete-time hazard model utilities."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV


def plot_calibration_reliability(
    result: Dict[str, object],
    n_bins: int = 10,
    results_dir: str = "results",
    filename: str = "reliability_calibration.png",
):
    """Plot reliability curves before vs after calibration from fit_hazard_model output."""
    if "p_test_cal" not in result:
        raise ValueError("Calibration outputs missing. Run fit_hazard_model(..., calibrate=True).")

    from src.plots import plot_reliability_curves

    return plot_reliability_curves(
        y_true=result["y_test"],
        p_uncal=result["p_test"],
        p_cal=result["p_test_cal"],
        n_bins=n_bins,
        results_dir=results_dir,
        filename=filename,
    )


def add_duration_bins(
    df: pd.DataFrame,
    duration_col: str = "t_in_spell",
    bins: Optional[List[int]] = None,
    labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Add duration bins used in the hazard model.

    Defaults match the notebook implementation.
    """
    out = df.copy()
    if bins is None:
        bins = [-1, 2, 5, 10, 20, 50, 200]
    if labels is None:
        labels = ["0-2", "3-5", "6-10", "11-20", "21-50", "51+"]

    out["dur_bin"] = pd.cut(
        out[duration_col],
        bins=bins,
        labels=labels,
        ordered=True,
    )
    return out


def time_based_split(df: pd.DataFrame, time_col: str = "year", train_end: int = 2010) -> Tuple[np.ndarray, np.ndarray]:
    """Return boolean masks for time-based train/test split."""
    train_mask = df[time_col] <= train_end
    test_mask = df[time_col] > train_end
    return train_mask.values, test_mask.values


def horizon_risk_from_hazard(
    model,
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    horizon: int = 3,
    duration_col: str = "t_in_spell",
    duration_bin_col: str = "dur_bin",
    bins: Optional[List[int]] = None,
    labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute H-year risk from a next-year hazard model.

    Uses covariates held fixed at time t and updates only duration bins.
    Returns a DataFrame with per-year hazards and the aggregated H-year risk.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if duration_bin_col not in feature_cols:
        raise ValueError("feature_cols must include the duration_bin_col")

    if bins is None:
        bins = [-1, 2, 5, 10, 20, 50, 200]
    if labels is None:
        labels = ["0-2", "3-5", "6-10", "11-20", "21-50", "51+"]

    base = df[list(feature_cols)].copy()
    hazards = []

    for k in range(1, horizon + 1):
        dur_k = df[duration_col] + k
        dur_bin_k = pd.cut(dur_k, bins=bins, labels=labels, ordered=True)
        Xk = base.copy()
        Xk[duration_bin_col] = dur_bin_k
        hk = model.predict_proba(Xk)[:, 1]
        hazards.append(hk)

    haz = np.vstack(hazards)  # shape: (H, N)
    risk_h = 1.0 - np.prod(1.0 - haz, axis=0)

    out = pd.DataFrame({f"hazard_t_plus_{k}": hazards[k - 1] for k in range(1, horizon + 1)}, index=df.index)
    out[f"risk_h{horizon}"] = risk_h
    return out


def _build_preprocessor(cat_cols: List[str], num_cols: List[str]) -> ColumnTransformer:
    cat_pre = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    num_pre = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    pre = ColumnTransformer(
        transformers=[
            ("cat", cat_pre, cat_cols),
            ("num", num_pre, num_cols),
        ],
        remainder="drop",
    )
    return pre


def evaluate_metrics(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """Compute ROC-AUC, PR-AUC, and Brier score."""
    out = {
        "pos_rate": float(np.mean(y_true)),
        "ROC_AUC": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else np.nan,
        "PR_AUC": float(average_precision_score(y_true, scores)) if len(np.unique(y_true)) > 1 else np.nan,
        "Brier": float(brier_score_loss(y_true, scores)),
    }
    return out


def fit_hazard_model(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    duration_bin_col: str = "dur_bin",
    target_col: str = "event",
    time_col: str = "year",
    include_country: bool = False,
    country_col: str = "country_key",
    train_end: int = 2010,
    calibrate: bool = True,
    class_weight: str = "balanced",
    max_iter: int = 5000,
) -> Dict[str, object]:
    """Fit a discrete-time hazard model and evaluate on a time-based split.

    Returns a dictionary with fitted model(s), predictions, and metrics.
    """
    use_cols = list(feature_cols) + [duration_bin_col]
    if include_country:
        use_cols.append(country_col)

    X = df[use_cols].copy()
    y = df[target_col].astype(int).values

    cat_cols = [duration_bin_col] + ([country_col] if include_country else [])
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = _build_preprocessor(cat_cols=cat_cols, num_cols=num_cols)
    clf = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=max_iter, class_weight=class_weight)),
    ])

    train_mask, test_mask = time_based_split(df, time_col=time_col, train_end=train_end)
    clf.fit(X[train_mask], y[train_mask])

    p_test = clf.predict_proba(X[test_mask])[:, 1]
    y_test = y[test_mask]
    metrics_uncal = evaluate_metrics(y_test, p_test)

    out: Dict[str, object] = {
        "model": clf,
        "X": X,
        "y": y,
        "train_mask": train_mask,
        "test_mask": test_mask,
        "p_test": p_test,
        "y_test": y_test,
        "metrics_uncal": metrics_uncal,
    }

    if calibrate:
        cal = CalibratedClassifierCV(clf, method="isotonic", cv=5)
        cal.fit(X[train_mask], y[train_mask])
        p_cal = cal.predict_proba(X[test_mask])[:, 1]
        metrics_cal = evaluate_metrics(y_test, p_cal)
        out.update({
            "calibrator": cal,
            "p_test_cal": p_cal,
            "metrics_cal": metrics_cal,
        })

    return out
