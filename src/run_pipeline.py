"""Offline-first democratic-breakdown research pipeline."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import uuid

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import ERT_RELEASE, load_ert, sha256_file
from src.case_audit import write_case_audit
from src.features import add_calendar_features, add_political_spells, construct_breakdown_outcomes
from src.model import (FittedRiskModel, PlattCalibrator, _pipeline, calibration_parameters,
                       country_bootstrap_metrics, metrics, recent_prevalence_predictions,
                       reliability_table, rolling_predictions, select_regularization)

SCHEMA_VERSION = "1.0.0"
OUTCOME_ID = "vdem_row_democracy_to_autocracy_within_3y_v1"
CROSSWALK_VERSION = "ert-country-text-id-v1"
POLITICAL_FEATURES = ["v2x_polyarchy"]
MODEL_FEATURES = ["years_in_democratic_spell", "v2x_polyarchy_available",
                  "v2x_polyarchy_change1", "v2x_polyarchy_mean3",
                  "v2x_polyarchy_missing", "v2x_polyarchy_carried_forward"]


def _commit() -> str:
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception: return "unknown"


def prepare_frame(path: Path, analysis_start: int = 1995) -> tuple[pd.DataFrame, pd.DataFrame]:
    ert = load_ert(path, min_year=1900, max_year=2024)
    # Preserve V-Dem's numeric identifier under an unambiguous name; the stable
    # text identifier is the modeling/crosswalk key.
    if "country_id" in ert: ert = ert.rename(columns={"country_id": "vdem_numeric_country_id"})
    ert = ert.rename(columns={"country_key": "country_id"})
    ert["iso3"] = ert["country_id"]
    if "country_name" not in ert: ert["country_name"] = ert.country_id
    # Spells and features are deliberately constructed before analysis-period selection.
    panel = add_political_spells(ert)
    panel = add_calendar_features(panel, POLITICAL_FEATURES, availability_lag_years=1, max_age_years=2)
    labelled = construct_breakdown_outcomes(panel, horizon=3, data_as_of_year=2024)
    labelled = labelled[labelled.year.ge(analysis_start)].copy()
    labelled = labelled.rename(columns={"year": "forecast_origin_year", "breakdown_within_horizon": "outcome"})
    return panel, labelled


def _cohort_summary(frame: pd.DataFrame, pred: pd.DataFrame) -> dict:
    event_rows = pred.loc[pred.outcome.eq(1)]
    return {"origin_year_min": int(pred.forecast_origin_year.min()), "origin_year_max": int(pred.forecast_origin_year.max()),
            "countries": int(pred.country_id.nunique()), "spells": int(pred[["country_id", "democratic_spell_id"]].drop_duplicates().shape[0]),
            "rows": len(pred), "positive_labels": int(pred.outcome.sum()),
            "distinct_events": int(event_rows[["country_id", "event_year"]].drop_duplicates().shape[0])}


def _score_table(panel: pd.DataFrame, model: FittedRiskModel, issued: str, run_id: str, commit: str, snapshot: str) -> pd.DataFrame:
    ref_year = int(panel.year.max())
    roster = panel.sort_values(["country_id", "year"]).groupby("country_id", as_index=False).tail(1)[["country_id", "iso3", "country_name"]]
    rows = roster.merge(panel.loc[panel.year.eq(ref_year)], on=["country_id", "iso3", "country_name"], how="left", validate="one_to_one")
    eligible = rows.political_state.eq("democracy")
    complete = rows[MODEL_FEATURES].notna().mean(axis=1)
    predictions = np.full(len(rows), np.nan)
    if eligible.any(): predictions[eligible.to_numpy()] = model.predict(rows.loc[eligible])
    reason = np.where(eligible, pd.NA, np.where(rows.year.isna(), "country_not_observed_at_reference_year",
                      np.where(rows.political_state.eq("autocracy"), "not_democratic_at_origin", "unknown_political_state")))
    result = pd.DataFrame({
        "schema_version": SCHEMA_VERSION, "model_type": "democratic_breakdown", "outcome_definition_id": OUTCOME_ID,
        "country_id": rows.country_id, "iso3": rows.iso3, "country_name": rows.country_name,
        "country_crosswalk_version": CROSSWALK_VERSION, "forecast_origin_year": ref_year,
        "forecast_issued_at": issued, "political_state_reference_year": ref_year,
        "forecast_horizon_years": 3, "target_end_year": ref_year + 3, "estimated_risk": predictions,
        "risk_interval": pd.NA, "eligibility_status": np.where(eligible, "eligible_retrospective", "ineligible"),
        "score_status": np.where(eligible, "scored_research_only", "unavailable"), "unavailable_reason": reason,
        "model_version": "breakdown-logit-v1", "run_id": run_id, "code_commit": commit,
        "data_snapshot_id": snapshot, "data_as_of": "2024-12-31", "feature_data_completeness": complete,
        "feature_max_age_years": rows[[c for c in rows if c.endswith("_age_years")]].max(axis=1),
        "staleness_status": np.where(rows.v2x_polyarchy_age_years.le(2), "within_assumption", "stale_or_missing"),
        "evaluation_cohort_id": "historical-2015-2021", "calibration_version": "platt-temporal-oos-v1",
    })
    return result


def run(input_path: Path, output_root: Path) -> Path:
    panel, labelled = prepare_frame(input_path)
    model_df = labelled.loc[labelled.evaluation_eligible & labelled.outcome.notna()].copy()
    model_df["outcome"] = model_df.outcome.astype(int)
    dev_years, cal_years, eval_years = range(2005, 2011), range(2011, 2015), range(2015, 2022)
    best_c, tuning = select_regularization(model_df, MODEL_FEATURES, dev_years)
    cal_pred = rolling_predictions(model_df, MODEL_FEATURES, cal_years, best_c)
    eval_rich = rolling_predictions(model_df, MODEL_FEATURES, eval_years, best_c)
    duration_pred = rolling_predictions(model_df, ["years_in_democratic_spell"], eval_years, best_c)
    score_pred = rolling_predictions(model_df, ["years_in_democratic_spell", "v2x_polyarchy_available"], eval_years, best_c)
    recent_pred = recent_prevalence_predictions(model_df, eval_years)
    train_prev = recent_pred.copy()
    historic = model_df[model_df.forecast_origin_year.lt(min(eval_years) - 3)].outcome.mean()
    train_prev["prediction"] = historic
    calibrator = None
    if len(cal_pred) and cal_pred.outcome.nunique() == 2:
        calibrator = PlattCalibrator().fit(cal_pred.prediction, cal_pred.outcome)
        eval_rich["prediction"] = calibrator.predict(eval_rich.prediction)

    comparisons, predictions = {}, {}
    for name, pred in {"regularized_logit": eval_rich, "duration_only": duration_pred,
                       "duration_plus_score": score_pred, "recent_prevalence": recent_pred,
                       "training_prevalence": train_prev}.items():
        m = metrics(pred.outcome, pred.prediction); intercept, slope = calibration_parameters(pred.outcome.to_numpy(), pred.prediction.to_numpy())
        comparisons[name] = {**m, "calibration_intercept": intercept, "calibration_slope": slope}
        predictions[name] = pred

    # Promotion rule is specified before comparison in config; enforce it mechanically.
    leading = min(comparisons, key=lambda x: comparisons[x]["log_loss"])
    status = "research-only"  # retrospective vintage and limited event support prohibit deployment claim.
    final_train = model_df[model_df.label_available_year.le(2024)]
    estimator = _pipeline(MODEL_FEATURES, best_c).fit(final_train[MODEL_FEATURES], final_train.outcome)
    fitted = FittedRiskModel(estimator, calibrator, MODEL_FEATURES)
    issued = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    run_id = f"run-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    out = output_root / run_id; out.mkdir(parents=True)
    commit, snapshot = _commit(), f"{ERT_RELEASE}-{sha256_file(input_path)[:12]}"
    scores = _score_table(panel, fitted, issued, run_id, commit, snapshot)
    evaluation = {"research_status": status, "leading_candidate": leading,
                  "cohort": _cohort_summary(model_df, eval_rich), "models": comparisons,
                  "fixed_prediction_country_bootstrap_95": country_bootstrap_metrics(eval_rich),
                  "overlapping_positive_labels": int(eval_rich.outcome.sum()),
                  "distinct_events": int(eval_rich.loc[eval_rich.outcome.eq(1), ["country_id", "event_year"]].drop_duplicates().shape[0]),
                  "non_overlapping_origin_sensitivity": metrics(
                      eval_rich.loc[eval_rich.forecast_origin_year.isin([2015, 2018, 2021]), "outcome"],
                      eval_rich.loc[eval_rich.forecast_origin_year.isin([2015, 2018, 2021]), "prediction"]),
                  "alternative_cutoff_sensitivity_2016_2021": metrics(
                      eval_rich.loc[eval_rich.forecast_origin_year.ge(2016), "outcome"],
                      eval_rich.loc[eval_rich.forecast_origin_year.ge(2016), "prediction"]),
                  "by_origin_year": {str(y): metrics(g.outcome, g.prediction) for y, g in eval_rich.groupby("forecast_origin_year")},
                  "feature_missingness": {f: float(model_df[f].isna().mean()) for f in MODEL_FEATURES},
                  "note": "Evaluation years were inspected by the original notebook and are not a newly untouched holdout."}
    (out / "metrics.json").write_text(json.dumps(evaluation, indent=2, allow_nan=True) + "\n")
    tuning.to_csv(out / "tuning.csv", index=False)
    eval_rich.to_csv(out / "evaluation_predictions.csv", index=False)
    reliability_table(eval_rich.outcome, eval_rich.prediction).to_csv(out / "reliability.csv", index=False)
    scores.to_csv(out / "predictions.csv", index=False)
    with (out / "model.pkl").open("wb") as handle: pickle.dump(fitted, handle)
    config = json.loads((ROOT / "config" / "model.json").read_text())
    (out / "configuration.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "feature_definitions.json").write_text(json.dumps({"features": MODEL_FEATURES, "availability": "Political values observed in t-1 are assumed available at t; this is not verified publication timing."}, indent=2) + "\n")
    (out / "input_manifest.json").write_text(json.dumps({"ert": {"path": str(input_path), "sha256": sha256_file(input_path), "snapshot_id": snapshot}, "code_commit": commit}, indent=2) + "\n")
    (out / "prediction_schema.json").write_text((ROOT / "schemas" / "predictions.schema.json").read_text())
    write_case_audit(panel, labelled, out / "case_audit")
    print(json.dumps({"run_dir": str(out), "status": status, "leading_candidate": leading, "metrics": comparisons}, indent=2))
    return out


def main():
    p = argparse.ArgumentParser(description="Retrospective three-year democratic-breakdown research pipeline")
    p.add_argument("--input", type=Path, default=ROOT / "notebooks/data/ert.csv")
    p.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    args = p.parse_args(); run(args.input, args.output_dir)

if __name__ == "__main__": main()
