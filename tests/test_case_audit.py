import pandas as pd

from src.case_audit import build_distinct_event_audit, build_positive_origin_audit, case_audit_summary
from src.features import add_calendar_features, add_political_spells, construct_breakdown_outcomes


def _frames():
    raw = pd.DataFrame({
        "country_id": ["AAA"] * 8,
        "country_name": ["Example"] * 8,
        "iso3": ["AAA"] * 8,
        "year": range(2000, 2008),
        "v2x_regime": [2, 2, 2, 1, 2, 2, 2, 2],
        "v2x_polyarchy": [.6, .58, .55, .4, .52, .55, .57, .59],
    })
    panel = add_calendar_features(add_political_spells(raw), ["v2x_polyarchy"])
    origins = construct_breakdown_outcomes(panel, horizon=3, data_as_of_year=2007).rename(
        columns={"year": "forecast_origin_year", "breakdown_within_horizon": "outcome"}
    )
    return panel, origins


def test_positive_origins_expose_overlapping_labels():
    _, origins = _frames()
    audit = build_positive_origin_audit(origins)
    assert audit.forecast_origin_year.tolist() == [2000, 2001, 2002]
    assert audit.years_until_event.tolist() == [3, 2, 1]
    assert audit.labels_for_same_event.eq(3).all()


def test_distinct_event_records_transition_and_recovery():
    panel, origins = _frames()
    events = build_distinct_event_audit(panel, origins)
    assert len(events) == 1
    event = events.iloc[0]
    assert (event.regime_before, event.regime_event_year) == (2, 1)
    assert event.first_recovery_year == 2004
    assert bool(event.temporary_exit_within_3y)
    assert not bool(event.positive_window_left_truncated)
    summary = case_audit_summary(events, origins)
    assert summary == {**summary, "positive_origin_labels": 3, "distinct_transition_events": 1}
