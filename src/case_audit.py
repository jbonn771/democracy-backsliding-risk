"""Case-level audits for the democratic-breakdown outcome.

These tables describe how the frozen V-Dem classification generates labels.
They are diagnostic, not causal narratives or adjudications that a coding is
substantively correct.
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from src.features import DEMOCRACY


def build_positive_origin_audit(origins: pd.DataFrame) -> pd.DataFrame:
    """Return every positive origin and expose dependence among horizon labels."""
    positive = origins.loc[origins["outcome"].eq(1)].copy()
    positive["years_until_event"] = positive["event_year"] - positive["forecast_origin_year"]
    positive["event_key"] = positive["country_id"].astype(str) + "-" + positive["event_year"].astype(str)
    positive["labels_for_same_event"] = positive.groupby("event_key")["event_key"].transform("size")
    columns = [
        "event_key", "country_id", "country_name", "forecast_origin_year", "event_year",
        "years_until_event", "v2x_regime", "v2x_polyarchy", "v2x_polyarchy_available",
        "years_in_democratic_spell", "spell_start_status", "label_status",
        "followup_complete", "administratively_matured", "evaluation_eligible",
        "labels_for_same_event",
    ]
    return positive[columns].sort_values(["event_year", "country_id", "forecast_origin_year"]).reset_index(drop=True)


def build_distinct_event_audit(panel: pd.DataFrame, origins: pd.DataFrame) -> pd.DataFrame:
    """Collapse overlapping positive origins to one row per observed transition."""
    positive = build_positive_origin_audit(origins)
    events = positive.groupby(["event_key", "country_id", "country_name", "event_year"], as_index=False).agg(
        first_positive_origin=("forecast_origin_year", "min"),
        last_positive_origin=("forecast_origin_year", "max"),
        overlapping_positive_labels=("forecast_origin_year", "size"),
        matured_positive_labels=("evaluation_eligible", "sum"),
        origin_spell_start_status=("spell_start_status", "first"),
    )
    events["positive_window_left_truncated"] = events["first_positive_origin"].gt(events["event_year"] - 3)
    lookup = panel.set_index(["country_id", "year"])
    country_event_counts = events.groupby("country_id")["event_key"].transform("size")
    rows = []
    max_year = int(panel["year"].max())
    for i, event in events.iterrows():
        country, year = event.country_id, int(event.event_year)
        before = lookup.loc[(country, year - 1)] if (country, year - 1) in lookup.index else None
        after = lookup.loc[(country, year)] if (country, year) in lookup.index else None
        recovery_year = next((future for future in range(year + 1, max_year + 1)
                              if (country, future) in lookup.index
                              and lookup.loc[(country, future), "v2x_regime"] in DEMOCRACY), None)
        rows.append({
            **event.to_dict(),
            "regime_before": None if before is None else before["v2x_regime"],
            "regime_event_year": None if after is None else after["v2x_regime"],
            "polyarchy_before": None if before is None else before.get("v2x_polyarchy"),
            "polyarchy_event_year": None if after is None else after.get("v2x_polyarchy"),
            "polyarchy_change_at_event": None if before is None or after is None else after.get("v2x_polyarchy") - before.get("v2x_polyarchy"),
            "first_recovery_year": recovery_year,
            "years_to_recovery": None if recovery_year is None else recovery_year - year,
            "temporary_exit_within_3y": recovery_year is not None and recovery_year <= year + 3,
            "event_at_data_boundary": year == max_year,
            "country_has_repeated_breakdowns": country_event_counts.iloc[i] > 1,
        })
    return pd.DataFrame(rows).sort_values(["event_year", "country_id"]).reset_index(drop=True)


def case_audit_summary(events: pd.DataFrame, origins: pd.DataFrame) -> dict:
    positive = origins.loc[origins.outcome.eq(1)]
    return {
        "positive_origin_labels": int(len(positive)),
        "distinct_transition_events": int(len(events)),
        "labels_per_distinct_event": float(len(positive) / len(events)),
        "events_with_three_overlapping_labels": int(events.overlapping_positive_labels.eq(3).sum()),
        "events_with_left_truncated_positive_windows": int(events.positive_window_left_truncated.sum()),
        "temporary_exits_recovering_within_3y": int(events.temporary_exit_within_3y.sum()),
        "events_at_data_boundary": int(events.event_at_data_boundary.sum()),
        "countries_with_repeated_breakdowns": int(events.loc[events.country_has_repeated_breakdowns, "country_id"].nunique()),
        "note": "Counts describe the frozen classification. They do not validate each case's substantive interpretation.",
    }


def write_case_audit(panel: pd.DataFrame, origins: pd.DataFrame, output_dir: Path | str) -> dict:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    positive = build_positive_origin_audit(origins)
    events = build_distinct_event_audit(panel, origins)
    summary = case_audit_summary(events, origins)
    positive.to_csv(output / "positive_origins.csv", index=False)
    events.to_csv(output / "distinct_events.csv", index=False)
    (output / "case_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
