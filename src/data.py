"""Data ingestion utilities for ERT and World Bank WDI.

The functions here download or load raw data, standardize keys, and return
clean pandas DataFrames. No hardcoded local paths are used; callers pass paths
or cache directories explicitly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
import requests

# The vendored file is pinned by checksum because the original commit did not
# preserve an upstream release tag. New acquisitions must set a versioned URL.
ERT_RELEASE = "vendored-ert-sha256-39c6ca133637"
ERT_URL = "https://raw.githubusercontent.com/vdeminstitute/ERT/master/inst/ert.csv"

DEFAULT_WB_INDICATORS: Dict[str, str] = {
    # core macro
    "gdp_growth": "NY.GDP.MKTP.KD.ZG",
    "inflation_cpi": "FP.CPI.TOTL.ZG",
    "unemployment": "SL.UEM.TOTL.ZS",
    "gdp_pc_current_usd": "NY.GDP.PCAP.CD",

    # inequality (sparse)
    "gini": "SI.POV.GINI",

    # debt stress
    "ext_debt_gni": "DT.DOD.DECT.GN.ZS",
    "debt_service_exports": "DT.TDS.DECT.EX.ZS",

    # living standards proxy
    "consumption_pc_growth": "NE.CON.PRVT.PC.KD.ZG",

    # structural / sociological proxies
    "resource_rents_gdp": "NY.GDP.TOTL.RT.ZS",
    "pop_total": "SP.POP.TOTL",
    "pop_urban": "SP.URB.TOTL",

    # youth population components (derive 15-24 share)
    "pop_1519_m": "SP.POP.1519.MA",
    "pop_1519_f": "SP.POP.1519.FE",
    "pop_2024_m": "SP.POP.2024.MA",
    "pop_2024_f": "SP.POP.2024.FE",

    # youth unemployment
    "youth_unemployment": "SL.UEM.1524.ZS",

    # migration (absolute; derive per-1,000)
    "net_migration": "SM.POP.NETM",
}

WB_BASE = "https://api.worldbank.org/v2"


def download_file(url: str, dest_path: Path | str, overwrite: bool = False, timeout: int = 120) -> Path:
    """Atomically download a file, rejecting empty/error responses."""
    dest = Path(dest_path)
    if dest.exists() and not overwrite:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)

    tmp = dest.with_name(dest.name + ".partial")
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            content_type = r.headers.get("content-type", "")
            if "text/html" in content_type:
                raise ValueError(f"unexpected HTML response from {url}")
            with tmp.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk: f.write(chunk)
        if not tmp.exists() or tmp.stat().st_size == 0:
            raise ValueError(f"empty download from {url}")
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def write_input_manifest(files: Iterable[Path | str], destination: Path | str, metadata: dict) -> Path:
    """Write checksums and provenance alongside immutable input snapshots."""
    dest = Path(destination); dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {**metadata, "files": [{"path": str(Path(p)), "bytes": Path(p).stat().st_size,
                                      "sha256": sha256_file(p)} for p in files]}
    tmp = dest.with_name(dest.name + ".partial")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, dest)
    return dest


def _pick_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    """Find a column by exact, case-insensitive, or normalized match."""
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]

    # normalized match (remove underscores and spaces)
    import re

    norm = {re.sub(r"[_\s]+", "", c.lower()): c for c in df.columns}
    for cand in candidates:
        key = re.sub(r"[_\s]+", "", cand.lower())
        if key in norm:
            return norm[key]

    return None


def load_ert(
    source: Optional[str | Path] = None,
    cache_dir: Optional[str | Path] = None,
    min_year: int = 1900,
    max_year: int = 2024,
) -> pd.DataFrame:
    """Load and standardize the V-Dem ERT dataset.

    Parameters
    ----------
    source:
        URL or local path. If None, uses the official ERT URL.
    cache_dir:
        Optional directory to cache the downloaded CSV.
    min_year, max_year:
        Year filter applied after loading.

    Returns
    -------
    pd.DataFrame
        ERT data with standardized columns: `country_key`, `year`, `v2x_regime`.
    """
    if source is None:
        source = ERT_URL

    source_str = str(source)
    if source_str.startswith("http"):
        if cache_dir is not None:
            path = Path(cache_dir) / "ert.csv"
            download_file(source_str, path, overwrite=False)
            df = pd.read_csv(path)
        else:
            df = pd.read_csv(source_str)
    else:
        df = pd.read_csv(source_str)

    # Standardize key columns
    col_year = _pick_col(df, ["year"])
    col_iso3 = _pick_col(df, ["country_text_id", "country_textid", "iso3", "iso3c", "country_code"])
    col_name = _pick_col(df, ["country_name", "country"])
    col_regime = _pick_col(df, ["v2x_regime"])

    if col_year is None or col_regime is None:
        raise ValueError("ERT missing required columns. Need at least: year, v2x_regime.")

    out = df.copy()
    out["year"] = pd.to_numeric(out[col_year], errors="coerce").astype("Int64")
    out["v2x_regime"] = pd.to_numeric(out[col_regime], errors="coerce")

    if col_iso3 is not None:
        out["country_key"] = out[col_iso3].astype(str).str.strip().str.upper()
    else:
        out["country_key"] = out[col_name].astype(str).str.strip().str.upper()

    out = out[(out["year"] >= min_year) & (out["year"] <= max_year)].copy()
    out = out[out["country_key"].notna() & (out["country_key"] != "")].copy()
    required = ["country_key", "year", "v2x_regime"]
    if out.duplicated(["country_key", "year"]).any():
        raise ValueError("ERT contains duplicate country-year keys")
    if out.empty or any(c not in out for c in required):
        raise ValueError("ERT schema validation failed")
    return out


def _wb_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "wb-panel-builder/1.0"})
    return s


def _raise_if_wb_error(payload, indicator_code: str) -> None:
    if (
        isinstance(payload, list)
        and len(payload) >= 1
        and isinstance(payload[0], dict)
        and "message" in payload[0]
    ):
        raise ValueError(f"World Bank API error for {indicator_code}: {payload[0]['message']}")


def wb_fetch_indicator(
    indicator_code: str,
    session: Optional[requests.Session] = None,
    per_page: int = 20000,
    sleep_s: float = 0.1,
    timeout: int = 120,
) -> pd.DataFrame:
    """Fetch a single WDI indicator as a DataFrame with columns [iso3, year, value]."""
    session = session or _wb_session()
    url = f"{WB_BASE}/country/all/indicator/{indicator_code}"
    page = 1
    out = []

    while True:
        params = {"format": "json", "per_page": per_page, "page": page}
        r = session.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()

        _raise_if_wb_error(data, indicator_code)

        if not isinstance(data, list) or len(data) < 2 or not isinstance(data[0], dict):
            raise ValueError(f"Unexpected response shape for {indicator_code}: {str(data)[:300]}")

        meta = data[0]
        records = data[1] or []

        for rec in records:
            iso3 = rec.get("countryiso3code")
            year = rec.get("date")
            val = rec.get("value")

            if not iso3 or year is None:
                continue
            try:
                y = int(year)
            except Exception:
                continue

            out.append((iso3.strip().upper(), y, val))

        pages = meta.get("pages")
        if pages is None:
            if not records:
                break
        else:
            try:
                pages = int(pages)
            except Exception:
                pages = page
            if page >= pages:
                break

        page += 1
        time.sleep(sleep_s)

    df = pd.DataFrame(out, columns=["iso3", "year", indicator_code])
    df[indicator_code] = pd.to_numeric(df[indicator_code], errors="coerce")
    return df


def build_wb_panel(
    indicators: Dict[str, str] = DEFAULT_WB_INDICATORS,
    sleep_s: float = 0.15,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """Fetch and merge WDI indicators into a single country-year panel."""
    session = session or _wb_session()
    frames = []

    for name, code in indicators.items():
        df = wb_fetch_indicator(code, session=session, sleep_s=sleep_s).rename(columns={code: name})
        frames.append(df)

    panel = frames[0]
    for df in frames[1:]:
        panel = panel.merge(df, on=["iso3", "year"], how="outer")

    panel = panel.sort_values(["iso3", "year"]).reset_index(drop=True)
    return panel


def add_wb_derived_features(wb: pd.DataFrame) -> pd.DataFrame:
    """Add derived features used in the model (shares, migration, inflation surprise)."""
    out = wb.copy()

    pop_cols = ["pop_1519_m", "pop_1519_f", "pop_2024_m", "pop_2024_f"]
    for c in pop_cols:
        if c not in out.columns:
            out[c] = np.nan

    if "pop_total" not in out.columns:
        out["pop_total"] = np.nan

    out["youth_pop_1524"] = out[pop_cols].sum(axis=1, min_count=1)
    out["youth_share_1524"] = 100.0 * out["youth_pop_1524"] / out["pop_total"]

    if "pop_urban" not in out.columns:
        out["pop_urban"] = np.nan
    out["urban_share"] = 100.0 * out["pop_urban"] / out["pop_total"]

    if "net_migration" not in out.columns:
        out["net_migration"] = np.nan
    out["net_migration_per_1000"] = 1000.0 * out["net_migration"] / out["pop_total"]

    # Inflation surprise: current inflation minus trailing 5-year mean (prior years only)
    if "inflation_cpi" in out.columns:
        out = out.sort_values(["iso3", "year"]).copy()
        out["inflation_trailing5"] = (
            out.groupby("iso3")["inflation_cpi"]
            .transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
        )
        out["inflation_surprise"] = out["inflation_cpi"] - out["inflation_trailing5"]

    return out


def load_wb_panel(
    indicators: Dict[str, str] = DEFAULT_WB_INDICATORS,
    derive_features: bool = True,
    sleep_s: float = 0.15,
) -> pd.DataFrame:
    """Convenience wrapper to fetch WDI indicators and add derived features."""
    wb = build_wb_panel(indicators=indicators, sleep_s=sleep_s)
    if derive_features:
        wb = add_wb_derived_features(wb)
    return wb
