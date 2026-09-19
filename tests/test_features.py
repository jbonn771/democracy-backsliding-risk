import numpy as np
import pandas as pd
import pytest
from src.features import add_calendar_features, add_political_spells, construct_breakdown_outcomes, validate_panel

def panel(states, years=None, country="AAA"):
    years = years or list(range(2000, 2000 + len(states)))
    return pd.DataFrame({"country_id": country, "year": years, "v2x_regime": states, "v2x_polyarchy": np.arange(len(states), dtype=float)})

def test_transition_recovery_and_temporary_exit_counts():
    p = add_political_spells(panel([2, 2, 1, 2, 2, 2, 2]))
    out = construct_breakdown_outcomes(p, 3, 2006)
    assert out.loc[out.year.eq(2000), "breakdown_within_horizon"].item() == 1
    assert out.loc[out.year.eq(2003), "democratic_spell_id"].item() == 2
    assert out.loc[out.year.eq(2003), "breakdown_within_horizon"].item() == 0

def test_missing_classification_is_unknown_not_negative():
    out = construct_breakdown_outcomes(add_political_spells(panel([2, np.nan, 2, 2])), 2, 2003)
    row = out[out.year.eq(2000)].iloc[0]
    assert pd.isna(row.breakdown_within_horizon) and row.label_status == "incomplete_followup"

def test_internal_gap_interrupts_followup_and_spell():
    p = add_political_spells(panel([2, 2], years=[2000, 2002]))
    out = construct_breakdown_outcomes(p, 2, 2002)
    assert out.iloc[0].label_status == "incomplete_followup"
    assert out.democratic_spell_id.tolist() == [1, 2]

def test_duplicate_and_horizon_validation():
    p = panel([2, 2]); dup = pd.concat([p, p.iloc[[0]]])
    with pytest.raises(ValueError, match="duplicate"): validate_panel(dup)
    with pytest.raises(ValueError, match="positive integer"): construct_breakdown_outcomes(p, 0)

def test_recent_event_not_evaluation_eligible_until_cohort_matures():
    out = construct_breakdown_outcomes(add_political_spells(panel([2, 1], years=[2023, 2024])), 3, 2024)
    assert out.iloc[0].breakdown_within_horizon == 1
    assert not out.iloc[0].evaluation_eligible

def test_single_country_calendar_features_do_not_cross_gap_or_use_current_value():
    p = panel([2, 2, 2], years=[2000, 2001, 2004])
    f = add_calendar_features(p, ["v2x_polyarchy"], availability_lag_years=1, max_age_years=1)
    assert f.loc[f.year.eq(2001), "v2x_polyarchy_available"].item() == 0
    assert pd.isna(f.loc[f.year.eq(2004), "v2x_polyarchy_available"].item())

def test_left_truncation_is_explicit():
    p = add_political_spells(panel([2, 2, 1, 2]))
    assert p.iloc[0].spell_start_status == "left_truncated_or_unknown"
    assert p.iloc[3].spell_start_status == "observed"
