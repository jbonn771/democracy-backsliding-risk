"""Artifact validation and loading helpers."""
from pathlib import Path
import json
import pickle
import pandas as pd

REQUIRED = ["schema_version", "model_type", "outcome_definition_id", "country_id", "iso3", "country_name",
"country_crosswalk_version", "forecast_origin_year", "forecast_issued_at", "political_state_reference_year",
"forecast_horizon_years", "target_end_year", "estimated_risk", "risk_interval", "eligibility_status", "score_status",
"unavailable_reason", "model_version", "run_id", "code_commit", "data_snapshot_id", "data_as_of",
"feature_data_completeness", "feature_max_age_years", "staleness_status", "evaluation_cohort_id", "calibration_version"]

def validate_predictions(frame: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED) - set(frame.columns))
    if missing: raise ValueError(f"missing prediction fields: {missing}")
    if not frame.model_type.eq("democratic_breakdown").all(): raise ValueError("invalid model_type")
    risk = frame.estimated_risk.dropna()
    if not risk.between(0, 1).all(): raise ValueError("estimated_risk outside [0,1]")
    unavailable = frame.score_status.eq("unavailable")
    if frame.loc[unavailable, "estimated_risk"].notna().any(): raise ValueError("unavailable rows must have null risk")
    if frame.loc[unavailable, "unavailable_reason"].isna().any(): raise ValueError("unavailable rows require a reason")

def load_run(path: Path | str):
    path = Path(path); predictions = pd.read_csv(path / "predictions.csv")
    validate_predictions(predictions)
    with (path / "model.pkl").open("rb") as handle: model = pickle.load(handle)
    return model, predictions, json.loads((path / "metrics.json").read_text())
