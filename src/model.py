"""Temporal three-year risk classification, calibration, and evaluation."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def temporal_train_mask(df: pd.DataFrame, prediction_year: int, horizon: int = 3, publication_delay: int = 0) -> np.ndarray:
    """Labels must have become available strictly before an OOS prediction year."""
    cutoff = prediction_year - publication_delay
    available = df.get("label_available_year", df["forecast_origin_year"] + horizon)
    return (available < cutoff).to_numpy()


def _pipeline(features: list[str], C: float = 1.0) -> Pipeline:
    prep = ColumnTransformer([("numeric", Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ]), features)], remainder="drop")
    return Pipeline([("preprocess", prep), ("classifier", LogisticRegression(
        C=C, l1_ratio=0, class_weight=None, solver="lbfgs", max_iter=2000, random_state=17
    ))])


def metrics(y: Iterable[int], p: Iterable[float]) -> dict[str, float]:
    y, p = np.asarray(y, int), np.clip(np.asarray(p, float), 1e-8, 1 - 1e-8)
    prevalence = float(y.mean())
    varied = len(np.unique(y)) == 2
    brier = float(brier_score_loss(y, p))
    return {
        "rows": int(len(y)), "positives": int(y.sum()), "prevalence": prevalence,
        "roc_auc": float(roc_auc_score(y, p)) if varied else math.nan,
        "average_precision": float(average_precision_score(y, p)) if varied else math.nan,
        "ap_prevalence_lift": float(average_precision_score(y, p) / prevalence) if varied and prevalence else math.nan,
        "brier_score": brier,
        "brier_skill_vs_prevalence": float(1 - brier / (prevalence * (1 - prevalence))) if 0 < prevalence < 1 else math.nan,
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "mean_predicted": float(p.mean()), "observed_incidence": prevalence,
    }


def calibration_parameters(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Logistic calibration intercept and slope (NaN when unsupported)."""
    if len(np.unique(y)) < 2:
        return math.nan, math.nan
    logits = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))[:, None]
    fit = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000).fit(logits, y)
    return float(fit.intercept_[0]), float(fit.coef_[0, 0])


class PlattCalibrator:
    def __init__(self): self.model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    def fit(self, p, y):
        p = np.asarray(p, dtype=float)
        x = np.log(np.clip(p, 1e-6, 1-1e-6) / np.clip(1-p, 1e-6, 1))[:, None]
        self.model.fit(x, y); return self
    def predict(self, p):
        p = np.asarray(p, dtype=float)
        x = np.log(np.clip(p, 1e-6, 1-1e-6) / np.clip(1-p, 1e-6, 1))[:, None]
        return self.model.predict_proba(x)[:, 1]


def rolling_predictions(df: pd.DataFrame, features: list[str], years: Iterable[int], C: float, horizon: int = 3) -> pd.DataFrame:
    """Generate calendar rolling-origin predictions without overlapping labels."""
    parts = []
    for year in years:
        test = df["forecast_origin_year"].eq(year)
        train = temporal_train_mask(df, year, horizon) & df["evaluation_eligible"].to_numpy()
        if test.sum() == 0 or train.sum() == 0 or df.loc[train, "outcome"].nunique() < 2:
            continue
        model = _pipeline(features, C).fit(df.loc[train, features], df.loc[train, "outcome"])
        part = df.loc[test, ["country_id", "forecast_origin_year", "outcome", "event_year", "democratic_spell_id"]].copy()
        part["prediction"] = model.predict_proba(df.loc[test, features])[:, 1]
        parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def recent_prevalence_predictions(df: pd.DataFrame, years: Iterable[int], lookback: int = 10) -> pd.DataFrame:
    parts = []
    for year in years:
        test = df["forecast_origin_year"].eq(year)
        train = temporal_train_mask(df, year) & df["forecast_origin_year"].ge(year - lookback - 3).to_numpy() & df["evaluation_eligible"].to_numpy()
        if not test.any() or not train.any(): continue
        part = df.loc[test, ["country_id", "forecast_origin_year", "outcome", "event_year", "democratic_spell_id"]].copy()
        part["prediction"] = float(df.loc[train, "outcome"].mean()); parts.append(part)
    return pd.concat(parts, ignore_index=True)


def select_regularization(df: pd.DataFrame, features: list[str], development_years: Iterable[int], candidates=(0.03, 0.1, 0.3, 1.0)) -> tuple[float, pd.DataFrame]:
    rows = []
    for C in candidates:
        pred = rolling_predictions(df, features, development_years, C)
        score = metrics(pred["outcome"], pred["prediction"]) if len(pred) else {"log_loss": math.inf}
        rows.append({"C": C, **score})
    table = pd.DataFrame(rows)
    return float(table.loc[table["log_loss"].idxmin(), "C"]), table


def country_bootstrap_metrics(pred: pd.DataFrame, draws: int = 300, seed: int = 17) -> dict[str, list[float]]:
    """Fixed-prediction intervals, resampling countries with full histories."""
    rng = np.random.default_rng(seed); countries = pred["country_id"].unique(); values = []
    for _ in range(draws):
        sample = rng.choice(countries, len(countries), replace=True)
        boot = pd.concat([pred[pred.country_id.eq(c)].assign(_rep=i) for i, c in enumerate(sample)])
        values.append(metrics(boot.outcome, boot.prediction))
    result = {}
    for key in ("roc_auc", "average_precision", "brier_score", "log_loss"):
        vals = np.asarray([x[key] for x in values], float)
        result[key] = [float(np.nanquantile(vals, .025)), float(np.nanquantile(vals, .975))]
    return result


def reliability_table(y, p, bins: int = 5) -> pd.DataFrame:
    frame = pd.DataFrame({"outcome": y, "prediction": p})
    frame["bin"] = pd.qcut(frame.prediction, q=min(bins, frame.prediction.nunique()), duplicates="drop")
    out = frame.groupby("bin", observed=True).agg(count=("outcome", "size"), predicted=("prediction", "mean"), observed=("outcome", "mean"), events=("outcome", "sum")).reset_index(drop=True)
    out["se"] = np.sqrt(out.observed * (1-out.observed) / out["count"])
    out["lower_95"] = (out.observed - 1.96*out.se).clip(0, 1); out["upper_95"] = (out.observed + 1.96*out.se).clip(0, 1)
    return out


@dataclass
class FittedRiskModel:
    estimator: Pipeline
    calibrator: PlattCalibrator | None
    features: list[str]
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        p = self.estimator.predict_proba(frame[self.features])[:, 1]
        return self.calibrator.predict(p) if self.calibrator else p
