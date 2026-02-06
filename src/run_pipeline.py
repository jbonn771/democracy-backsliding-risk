"""End-to-end pipeline: load data, build features, fit model, and save plots."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_ert, load_wb_panel, DEFAULT_WB_INDICATORS
from src.features import (
    construct_risk_set,
    add_lagged_features,
    add_change_features,
    add_rolling_features,
)
from src.model import add_duration_bins, fit_hazard_model, plot_calibration_reliability
from src.plots import (
    compute_km_table,
    plot_km_curve,
    plot_baseline_hazard,
    simulate_hazard_curves,
    plot_hazard_simulation,
)


BASE_FEATURES = [
    "inflation_surprise",
    "unemployment",
    "gdp_growth",
    "consumption_pc_growth",
    "gini",
    "debt_service_exports",
    "ext_debt_gni",
    "resource_rents_gdp",
    "urban_share",
    "youth_share_1524",
    "youth_unemployment",
    "net_migration_per_1000",
]

PRED_COLS = [
    "inflation_surprise_rollmean5", "inflation_surprise_chg1", "inflation_surprise_rollstd5",
    "unemployment_rollmean5", "unemployment_chg1",
    "consumption_pc_growth_rollmean5", "consumption_pc_growth_chg1",
    "gini_rollmean5",
    "debt_service_exports_rollmean5", "debt_service_exports_chg1",
    "ext_debt_gni_rollmean5",
    "resource_rents_gdp_rollmean5",
    "urban_share_rollmean5",
    "youth_share_1524_rollmean5",
    "youth_unemployment_rollmean5",
    "net_migration_per_1000_rollmean5",
]

SPARSE = [
    "gini_rollmean5",
    "ext_debt_gni_rollmean5",
    "debt_service_exports_rollmean5",
    "resource_rents_gdp_rollmean5",
    "youth_unemployment_rollmean5",
    "net_migration_per_1000_rollmean5",
]


def _build_feature_frame(risk: pd.DataFrame) -> pd.DataFrame:
    df = risk.copy()

    # Ensure base feature columns exist
    for f in BASE_FEATURES:
        if f not in df.columns:
            df[f] = np.nan

    df = add_lagged_features(df, group_col="country_key", time_col="year", cols=BASE_FEATURES, lags=(1, 2))
    df = add_change_features(df, cols=BASE_FEATURES, lag1=1, lag2=2)
    df = add_rolling_features(df, group_col="country_key", time_col="year", cols=BASE_FEATURES, window=5, min_periods=3)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run democracy backsliding risk pipeline.")
    parser.add_argument("--horizon", type=int, default=3, help="Outcome horizon H (default: 3)")
    parser.add_argument(
        "--outcome",
        type=str,
        default="within_horizon",
        choices=["within_horizon", "next_year"],
        help="Outcome definition (default: within_horizon)",
    )
    parser.add_argument("--train-end", type=int, default=2010, help="Train/test split year (default: 2010)")
    parser.add_argument("--include-country", action="store_true", help="Include country fixed effects")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory for output plots")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory for cached data")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Load ERT (use local cache if present)
    ert_path = data_dir / "ert.csv"
    if ert_path.exists():
        ert = load_ert(source=str(ert_path))
    else:
        ert = load_ert(cache_dir=str(data_dir))

    # Load WB data and derived features
    wb = load_wb_panel(DEFAULT_WB_INDICATORS)
    wb = wb.rename(columns={"iso3": "country_key"})

    # Merge ERT + WB
    merged = ert.merge(wb, on=["country_key", "year"], how="left")

    # Risk set and spells
    risk, spell_summary = construct_risk_set(
        merged, horizon=args.horizon, outcome=args.outcome
    )

    # Feature engineering
    risk = _build_feature_frame(risk)

    # Duration bins
    haz = add_duration_bins(risk)

    # Missingness flags for sparse predictors + limited forward fill
    haz = haz.sort_values(["country_key", "year"]).copy()
    for col in SPARSE:
        if col not in haz.columns:
            haz[col] = np.nan
        haz[col + "_missing"] = haz[col].isna().astype(int)
        haz[col] = haz.groupby("country_key")[col].transform(lambda s: s.ffill(limit=5))

    final_pred = PRED_COLS + [c + "_missing" for c in SPARSE]

    # Model frame
    keep_cols = ["country_key", "year", "event", "t_in_spell", "dur_bin"] + final_pred
    haz_model = haz[keep_cols].replace([np.inf, -np.inf], np.nan)
    haz_model = haz_model[haz_model["dur_bin"].notna()].copy()

    # Fit model
    res = fit_hazard_model(
        haz_model,
        feature_cols=final_pred,
        duration_bin_col="dur_bin",
        target_col="event",
        time_col="year",
        include_country=args.include_country,
        country_col="country_key",
        train_end=args.train_end,
        calibrate=True,
    )

    # Plots
    km_table = compute_km_table(spell_summary)
    plot_km_curve(km_table, results_dir=args.results_dir, filename="km_curves.png")
    plot_baseline_hazard(km_table, results_dir=args.results_dir, filename="baseline_hazard.png")

    plot_calibration_reliability(
        res, n_bins=10, results_dir=args.results_dir, filename="reliability_calibration.png"
    )

    # Hazard simulation (uses calibrated model if available)
    model_for_sim = res.get("calibrator", res["model"])
    sim_ref = haz_model[final_pred + ["dur_bin"]].copy()
    sim = simulate_hazard_curves(
        model_for_sim,
        reference_df=sim_ref,
        var="unemployment_chg1",
        values=[-2, -1, 0, 1, 2],
        duration_bins=["0-2", "3-5", "6-10", "11-20", "21-50", "51+"],
    )
    plot_hazard_simulation(
        sim,
        var="unemployment_chg1",
        results_dir=args.results_dir,
        filename="hazard_simulations.png",
        title="Effect of unemployment change (Delta1) across duration bins",
    )

    # Print summary metrics
    print("Uncalibrated metrics:", res["metrics_uncal"])
    if "metrics_cal" in res:
        print("Calibrated metrics:", res["metrics_cal"])


if __name__ == "__main__":
    main()
