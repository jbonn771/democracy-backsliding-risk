"""Plotting utilities for survival curves and hazard simulations."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def compute_km_table(
    spell_summary: pd.DataFrame,
    duration_col: str = "duration_years",
    event_col: str = "event",
) -> pd.DataFrame:
    """Compute a Kaplan-Meier table from spell-level data."""
    km = spell_summary.copy()
    max_d = int(km[duration_col].max())

    rows = []
    surv = 1.0
    for d in range(1, max_d + 1):
        at_risk = int((km[duration_col] >= d).sum())
        events = int(((km[duration_col] == d) & (km[event_col] == 1)).sum())
        if at_risk == 0:
            break
        hazard = events / at_risk
        surv *= (1 - hazard)
        rows.append((d, at_risk, events, hazard, surv))

    return pd.DataFrame(rows, columns=["duration_years", "at_risk", "events", "hazard", "survival"])


def plot_km_curve(
    km_table: pd.DataFrame,
    results_dir: str | Path = "results",
    filename: str = "km_curves.png",
) -> Path:
    """Plot Kaplan-Meier survival curve and save to results directory."""
    out_dir = _ensure_dir(results_dir)
    out_path = out_dir / filename

    plt.figure(figsize=(7, 4))
    plt.plot(km_table["duration_years"], km_table["survival"])
    plt.xlabel("Years into democratic spell")
    plt.ylabel("Survival (no autocratization onset yet)")
    plt.title("Kaplan-Meier survival of democracies (ERT-based)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_baseline_hazard(
    km_table: pd.DataFrame,
    results_dir: str | Path = "results",
    filename: str = "baseline_hazard.png",
) -> Path:
    """Plot baseline hazard by duration and save to results directory."""
    out_dir = _ensure_dir(results_dir)
    out_path = out_dir / filename

    plt.figure(figsize=(7, 4))
    plt.plot(km_table["duration_years"], km_table["hazard"])
    plt.xlabel("Years into democratic spell")
    plt.ylabel("Hazard (onset probability in that year)")
    plt.title("Baseline hazard by duration (unconditional)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def _reference_row(reference_df: pd.DataFrame, duration_bin_col: str) -> dict:
    num_cols = reference_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in reference_df.columns if c not in num_cols]

    row = {}
    if num_cols:
        row.update(reference_df[num_cols].median().to_dict())
    for c in cat_cols:
        if c == duration_bin_col:
            continue
        series = reference_df[c].dropna()
        if len(series) > 0:
            row[c] = series.mode().iloc[0]
    return row


def simulate_hazard_curves(
    model,
    reference_df: pd.DataFrame,
    var: str,
    values: Iterable[float],
    duration_bins: Iterable[str],
    duration_bin_col: str = "dur_bin",
) -> pd.DataFrame:
    """Simulate predicted hazards across duration bins for a single variable."""
    base = _reference_row(reference_df, duration_bin_col=duration_bin_col)

    rows = []
    for d in duration_bins:
        for v in values:
            row = base.copy()
            row[duration_bin_col] = d
            row[var] = v
            rows.append(row)

    sim = pd.DataFrame(rows)
    sim["pred_hazard"] = model.predict_proba(sim)[:, 1]
    return sim


def plot_hazard_simulation(
    sim: pd.DataFrame,
    var: str,
    results_dir: str | Path = "results",
    filename: str = "hazard_simulations.png",
    title: str = "Hazard simulation across duration bins",
) -> Path:
    """Plot simulated hazard curves and save to results directory."""
    out_dir = _ensure_dir(results_dir)
    out_path = out_dir / filename

    plt.figure(figsize=(8, 4))
    for v in sorted(sim[var].unique()):
        tmp = sim[sim[var] == v]
        plt.plot(tmp["dur_bin"], tmp["pred_hazard"], marker="o", label=f"{var}={v}")

    plt.xlabel("Years into democratic spell (bin)")
    plt.ylabel("Predicted hazard (this year)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def _reliability_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Compute a simple reliability table with equal-width bins."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins, right=True) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        rows.append({
            "bin": b,
            "count": int(mask.sum()),
            "pred_mean": float(np.mean(y_prob[mask])),
            "obs_rate": float(np.mean(y_true[mask])),
        })
    return pd.DataFrame(rows)


def plot_reliability_curves(
    y_true: np.ndarray,
    p_uncal: np.ndarray,
    p_cal: np.ndarray,
    n_bins: int = 10,
    results_dir: str | Path = "results",
    filename: str = "reliability_calibration.png",
) -> Path:
    """Plot reliability curves before vs after calibration."""
    out_dir = _ensure_dir(results_dir)
    out_path = out_dir / filename

    df_uncal = _reliability_table(y_true, p_uncal, n_bins=n_bins)
    df_cal = _reliability_table(y_true, p_cal, n_bins=n_bins)

    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Perfect calibration")
    plt.plot(df_uncal["pred_mean"], df_uncal["obs_rate"], marker="o", label="Uncalibrated")
    plt.plot(df_cal["pred_mean"], df_cal["obs_rate"], marker="o", label="Calibrated")
    plt.xlabel("Predicted probability (bin mean)")
    plt.ylabel("Observed event rate")
    plt.title(f"Reliability plot (n_bins={n_bins})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path
