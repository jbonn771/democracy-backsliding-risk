"""Calendar-safe panel, spell, feature, and breakdown outcome construction.

The estimand is a *regime-category transition*, not autocratization/erosion
onset.  All functions are label-free unless their name explicitly mentions
outcomes, so the same feature builder can score the latest eligible origin.
"""
from __future__ import annotations

from typing import Iterable
import numpy as np
import pandas as pd

DEMOCRACY = frozenset({2, 3})
AUTOCRACY = frozenset({0, 1})


def validate_panel(df: pd.DataFrame, country_col: str = "country_id", year_col: str = "year") -> None:
    """Reject missing keys, duplicate country-years, and non-integral years."""
    missing = [c for c in (country_col, year_col) if c not in df]
    if missing:
        raise ValueError(f"panel missing required columns: {missing}")
    if df[[country_col, year_col]].isna().any().any():
        raise ValueError("country-year keys may not be missing")
    years = pd.to_numeric(df[year_col], errors="coerce")
    if years.isna().any() or not np.equal(years, np.floor(years)).all():
        raise ValueError("years must be integral")
    dup = df.duplicated([country_col, year_col], keep=False)
    if dup.any():
        keys = df.loc[dup, [country_col, year_col]].head().to_dict("records")
        raise ValueError(f"duplicate country-years: {keys}")


def add_political_spells(
    df: pd.DataFrame,
    country_col: str = "country_id",
    year_col: str = "year",
    regime_col: str = "v2x_regime",
) -> pd.DataFrame:
    """Identify democratic spells on the complete political panel.

    Unknown states and calendar gaps interrupt spells. A spell that begins at
    a country's first observed democratic row is marked left-truncated unless
    an observed non-democratic prior year establishes its start.
    """
    validate_panel(df, country_col, year_col)
    out = df.sort_values([country_col, year_col]).copy()
    valid = out[regime_col].isin(DEMOCRACY | AUTOCRACY)
    out["political_state"] = np.select(
        [out[regime_col].isin(DEMOCRACY), out[regime_col].isin(AUTOCRACY)],
        ["democracy", "autocracy"], default="unknown"
    )
    prev_year = out.groupby(country_col)[year_col].shift()
    prev_state = out.groupby(country_col)["political_state"].shift()
    contiguous = out[year_col].eq(prev_year + 1)
    start = out["political_state"].eq("democracy") & (~contiguous | prev_state.ne("democracy"))
    out["democratic_spell_id"] = start.groupby(out[country_col]).cumsum().where(
        out["political_state"].eq("democracy")
    ).astype("Int64")
    first = out.groupby(country_col).cumcount().eq(0)
    unknown_start = start & (first | ~contiguous | prev_state.eq("unknown"))
    truncated_ids = set(zip(out.loc[unknown_start, country_col], out.loc[unknown_start, "democratic_spell_id"]))
    out["spell_start_status"] = pd.NA
    dem = out["political_state"].eq("democracy")
    out.loc[dem, "spell_start_status"] = [
        "left_truncated_or_unknown" if (c, s) in truncated_ids else "observed"
        for c, s in zip(out.loc[dem, country_col], out.loc[dem, "democratic_spell_id"])
    ]
    out["years_in_democratic_spell"] = (
        out.loc[dem].groupby([country_col, "democratic_spell_id"]).cumcount() + 1
    ).reindex(out.index).astype("Int64")
    out["regime_valid"] = valid
    return out


def construct_breakdown_outcomes(
    panel: pd.DataFrame,
    horizon: int = 3,
    data_as_of_year: int | None = None,
    country_col: str = "country_id",
    year_col: str = "year",
    regime_col: str = "v2x_regime",
) -> pd.DataFrame:
    """Create explicit H-year first-transition labels for democratic origins.

    Evaluation eligibility requires an administratively matured origin cohort
    (`t + H <= data_as_of_year`) even when an event happens early. Unknown or
    absent required follow-up before the first event is never a negative.
    """
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    validate_panel(panel, country_col, year_col)
    if data_as_of_year is None:
        data_as_of_year = int(panel[year_col].max())
    lookup = panel.set_index([country_col, year_col])[regime_col].to_dict()
    origins = panel.loc[panel[regime_col].isin(DEMOCRACY)].copy()
    rows = []
    for idx, row in origins.iterrows():
        country, year = row[country_col], int(row[year_col])
        states, event_year = [], None
        for future_year in range(year + 1, year + horizon + 1):
            state = lookup.get((country, future_year), np.nan)
            states.append(state)
            if event_year is None and state in AUTOCRACY:
                event_year = future_year
        pre_event = states if event_year is None else states[: event_year - year]
        complete_to_event = all(x in (DEMOCRACY | AUTOCRACY) for x in pre_event)
        full_followup = all(x in (DEMOCRACY | AUTOCRACY) for x in states)
        matured = year + horizon <= data_as_of_year
        if event_year is not None and complete_to_event:
            label, status = 1, "event_observed"
        elif full_followup:
            label, status = 0, "complete_non_event"
        else:
            label, status = pd.NA, "incomplete_followup"
        rows.append((idx, label, status, event_year, full_followup, matured))
    meta = pd.DataFrame(rows, columns=["_idx", "breakdown_within_horizon", "label_status", "event_year", "followup_complete", "administratively_matured"]).set_index("_idx")
    origins = origins.join(meta)
    origins["breakdown_within_horizon"] = origins["breakdown_within_horizon"].astype("Int64")
    origins["event_year"] = origins["event_year"].astype("Int64")
    origins["forecast_horizon_years"] = horizon
    origins["target_end_year"] = origins[year_col] + horizon
    origins["label_available_year"] = origins["target_end_year"]
    origins["evaluation_eligible"] = origins["administratively_matured"] & origins["breakdown_within_horizon"].notna()
    return origins


def add_calendar_features(
    panel: pd.DataFrame,
    value_cols: Iterable[str],
    availability_lag_years: int = 1,
    max_age_years: int = 3,
    country_col: str = "country_id",
    year_col: str = "year",
) -> pd.DataFrame:
    """Build lagged/rolling features before risk-set selection.

    The lag is an explicit availability *assumption*, not evidence of actual
    publication timing. Filling is bounded by elapsed calendar time, and every
    feature carries age, missingness, and imputation flags.
    """
    validate_panel(panel, country_col, year_col)
    if availability_lag_years < 0 or max_age_years < 0:
        raise ValueError("availability lag and max age must be nonnegative")
    out = panel.sort_values([country_col, year_col]).copy()
    # Reindex each country to a calendar grid so shift/rolling cannot cross gaps.
    frames = []
    for country, group in out.groupby(country_col, sort=False):
        years = pd.RangeIndex(int(group[year_col].min()), int(group[year_col].max()) + 1)
        g = group.set_index(year_col).reindex(years)
        g[country_col] = country
        g[year_col] = g.index
        g["_original_row"] = g.index.isin(group[year_col])
        for col in value_cols:
            if col not in g:
                g[col] = np.nan
            observed_year = pd.Series(np.where(g[col].notna(), g.index, np.nan), index=g.index).ffill()
            available = g[col].shift(availability_lag_years)
            available_obs_year = observed_year.shift(availability_lag_years)
            age = g.index.to_series() - available_obs_year
            filled = available.ffill().where(age <= max_age_years)
            g[f"{col}_available"] = filled
            g[f"{col}_age_years"] = age.where(filled.notna())
            g[f"{col}_missing"] = filled.isna().astype(int)
            g[f"{col}_carried_forward"] = (filled.notna() & available.isna()).astype(int)
            g[f"{col}_change1"] = filled - filled.shift(1)
            g[f"{col}_mean3"] = filled.shift(1).rolling(3, min_periods=2).mean()
        frames.append(g.loc[g["_original_row"]].drop(columns="_original_row"))
    return pd.concat(frames, ignore_index=True).sort_values([country_col, year_col]).reset_index(drop=True)


# Backward-compatible aliases intentionally fail toward the explicit estimand.
def construct_risk_set(df: pd.DataFrame, horizon: int = 3, **kwargs):
    country_col = kwargs.get("country_col", "country_key")
    out = df.rename(columns={country_col: "country_id"}) if country_col != "country_id" else df
    spells = add_political_spells(out)
    risk = construct_breakdown_outcomes(spells, horizon=horizon)
    return risk, pd.DataFrame()
