"""Feature engineering utilities for the democracy backsliding hazard model."""

from __future__ import annotations

from typing import Iterable, List, Tuple, Optional
import pandas as pd
import numpy as np


def construct_risk_set(
    df: pd.DataFrame,
    country_col: str = "country_key",
    year_col: str = "year",
    regime_col: str = "v2x_regime",
    democracy_threshold: int = 2,
    horizon: int = 3,
    outcome: str = "within_horizon",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Construct the democratic risk set and spell summary.

    Risk set definition follows the notebook:
    - Keep democracy-years with known next-year regime outcomes.
    - Event = autocratization in the next year (or within horizon if specified).
    - Spells break on year gaps or immediately after an event.

    Multi-year horizon logic:
    - y=1 if autocratization occurs in [t+1, t+H] within observed years.
    - y=0 if full H years are observed with no autocratization.
    - Row dropped if the panel ends before H and no event occurs (right-censored).

    Returns
    -------
    risk : pd.DataFrame
        Risk set with `event`, `t_in_spell`, and `spell_id` columns.
    spell_summary : pd.DataFrame
        One row per democratic spell with duration and event indicator.
    """
    if outcome not in {"next_year", "within_horizon"}:
        raise ValueError("outcome must be 'next_year' or 'within_horizon'")

    out = df.sort_values([country_col, year_col]).copy()

    out["is_democracy"] = out[regime_col].ge(democracy_threshold).astype("Int64")
    out["is_democracy_next_year"] = out.groupby(country_col)["is_democracy"].shift(-1)

    out["event_next_year"] = (
        (out["is_democracy"] == 1) & (out["is_democracy_next_year"] == 0)
    ).astype("Int64")
    # Backwards-compatible alias
    out["y_autocratization_next_year"] = out["event_next_year"]

    if horizon is None or horizon < 1:
        raise ValueError("horizon must be >= 1")

    if horizon == 1:
        out["event_horizon"] = out["event_next_year"]
    else:
        out["event_horizon"] = (
            out.groupby(country_col, group_keys=False)
            .apply(lambda g: _compute_horizon_event(g, horizon, year_col, "is_democracy"))
            .astype("Int64")
        )
    out[f"y_autocratization_within_{horizon}y"] = out["event_horizon"]

    risk = out[out["is_democracy"] == 1].copy()
    risk = risk.sort_values([country_col, year_col]).copy()

    # spells: break on year gaps or after an event
    risk["year_gap"] = risk.groupby(country_col)[year_col].diff().fillna(1)
    risk["new_spell"] = (risk["year_gap"] != 1).astype(int)

    risk["post_event_break"] = (
        risk.groupby(country_col)["event_next_year"].shift(1).fillna(0).astype(int)
    )
    risk["spell_break"] = ((risk["new_spell"] == 1) | (risk["post_event_break"] == 1)).astype(int)

    risk["spell_id"] = risk.groupby(country_col)["spell_break"].cumsum()
    risk["t_in_spell"] = risk.groupby([country_col, "spell_id"]).cumcount()

    if outcome == "next_year":
        risk = risk[risk["is_democracy_next_year"].notna()].copy()
        risk["event"] = risk["event_next_year"].astype("int64")
    else:
        risk = risk[risk["event_horizon"].notna()].copy()
        risk["event"] = risk["event_horizon"].astype("int64")

    spell_summary = (
        risk.groupby([country_col, "spell_id"])
        .agg(
            start_year=(year_col, "min"),
            end_year=(year_col, "max"),
            duration=("t_in_spell", "max"),
            event=("event", "max"),
        )
        .reset_index()
    )
    spell_summary["duration_years"] = spell_summary["duration"] + 1

    return risk, spell_summary


def _compute_horizon_event(
    group: pd.DataFrame,
    horizon: int,
    year_col: str,
    is_dem_col: str,
) -> pd.Series:
    """Compute event within horizon for one country group.

    Returns Int64 series with values {1, 0, <NA>}.
    """
    years = group[year_col].to_numpy()
    is_dem = group[is_dem_col].to_numpy()
    year_to_idx = {int(y): i for i, y in enumerate(years) if pd.notna(y)}

    out = np.full(len(group), pd.NA, dtype="object")

    for i, y in enumerate(years):
        if pd.isna(is_dem[i]) or is_dem[i] != 1:
            continue

        future_years = [int(y) + k for k in range(1, horizon + 1)]
        observed = []
        event_found = False

        for fy in future_years:
            idx = year_to_idx.get(fy)
            if idx is None:
                continue
            observed.append(fy)
            if is_dem[idx] == 0:
                event_found = True
                break

        if event_found:
            out[i] = 1
        else:
            if len(observed) == horizon:
                out[i] = 0
            else:
                out[i] = pd.NA

    return pd.Series(out, index=group.index, dtype="Int64")


def add_lagged_features(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    cols: Iterable[str],
    lags: Iterable[int] = (1, 2),
) -> pd.DataFrame:
    """Add lagged features by group, avoiding temporal leakage."""
    out = df.sort_values([group_col, time_col]).copy()
    for col in cols:
        for lag in lags:
            out[f"{col}_lag{lag}"] = out.groupby(group_col)[col].shift(lag)
    return out


def add_change_features(
    df: pd.DataFrame,
    cols: Iterable[str],
    lag1: int = 1,
    lag2: int = 2,
) -> pd.DataFrame:
    """Add simple change features based on lagged values (lag1 - lag2)."""
    out = df.copy()
    for col in cols:
        out[f"{col}_chg1"] = out[f"{col}_lag{lag1}"] - out[f"{col}_lag{lag2}"]
    return out


def add_rolling_features(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    cols: Iterable[str],
    window: int = 5,
    min_periods: int = 3,
) -> pd.DataFrame:
    """Add rolling mean/std and acceleration features, using only past data.

    The rolling window is computed on lagged values to avoid leakage.
    """
    out = df.sort_values([group_col, time_col]).copy()
    for col in cols:
        lag1 = out.groupby(group_col)[col].shift(1)
        roll_mean = lag1.groupby(out[group_col]).transform(
            lambda s: s.rolling(window, min_periods=min_periods).mean()
        )
        roll_std = lag1.groupby(out[group_col]).transform(
            lambda s: s.rolling(window, min_periods=min_periods).std()
        )

        out[f"{col}_rollmean{window}"] = roll_mean
        out[f"{col}_rollstd{window}"] = roll_std
        out[f"{col}_accel"] = lag1 - roll_mean
    return out


def ensure_no_leakage(df: pd.DataFrame, time_col: str, feature_cols: List[str]) -> None:
    """Basic guard to ensure feature columns are not forward-looking.

    This is a lightweight check: it raises if any feature column is identical
    to the target year column (a common leakage error).
    """
    for col in feature_cols:
        if col == time_col:
            raise ValueError("Feature columns must not include the time index column.")
