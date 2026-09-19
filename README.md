# Democratic Breakdown Risk — retrospective research pipeline

> **Outcome boundary:** this repository estimates a V-Dem Regimes of the World
> **democratic-to-autocratic category transition**. It does **not** estimate the
> beginning of democratic erosion or autocratization. A separate model should
> address erosion onset.

The unit is a country-year forecast origin *t*. Countries with `v2x_regime` 2
or 3 at *t* are eligible; the outcome is the first classification of 0 or 1 in
*t+1…t+3*. A temporary exit counts. A return to democracy creates a new spell.
Unknown classifications never count as autocracy, and already-autocratic
country-years receive a null score—not an artificial zero.

## Research status

**Research-only.** These are retrospective predictions based on a current
vintage, not forecasts that were issued historically. `forecast_origin_year`,
`political_state_reference_year`, and the actual artifact
`forecast_issued_at` are separate fields. The one-year political-data lag is
an availability assumption, not proof that V-Dem's value for year *t-1* was
public at issuance. Without archived V-Dem/WDI vintages, the evaluation is not
a real-time backtest and the artifacts are not deployment-ready.

Promotion criteria were frozen in `config/model.json` before comparison:
valid temporal/data boundaries, offline reproducibility, useful improvement
over simple baselines on proper scores and average precision, acceptable
calibration, adequate country-aware uncertainty/coverage, and stability over
evaluation origins. No accuracy or arbitrary AUC threshold is used.

## Reproducible run

Python 3.11–3.13 and exact direct dependencies are pinned.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pytest -q
python src/run_pipeline.py --input notebooks/data/ert.csv --output-dir artifacts
```

The offline command needs no network. Each run directory contains the fitted
preprocessing/model/calibration pickle, configuration, feature definitions,
predictions, JSON Schema, temporal evaluation predictions, reliability bins,
tuning results, metrics, and checksummed input manifest. Load and validate a
round trip with `src.artifacts.load_run(path)`.

## Scientific design

1. The complete 1900–2024 political panel is validated for unique keys.
2. Political spells are formed before the 1995 modeling boundary. Calendar
   gaps and unknown states interrupt spells; uncertain starts are marked left
   truncated. Re-democratization starts a new spell.
3. Calendar grids are inserted before lag/change/rolling computation. Carries
   are limited by elapsed calendar years, and age/missing/carried-forward
   fields are retained. Learned imputation and scaling live in the sklearn
   pipeline and are fitted only on each permitted training window.
4. Labels retain status, completeness, event year, horizon end, and label
   availability. Ordinary binary evaluation requires the whole origin cohort
   to be administratively mature even if its event occurred early.
5. A regularized logistic **three-year risk classifier** starts with
   `class_weight=None`. Temporal development folds select regularization.
   Chronological out-of-sample development predictions support low-parameter
   Platt calibration. It is not an annual hazard and its probabilities are
   never compounded.
6. Calendar rolling-origin evaluation purges labels whose windows overlap a
   prediction origin. Evaluation compares training prevalence, recent-history
   prevalence, duration-only, duration-plus-available political score, and the
   richer candidate on identical years. The 2015–2021 period was previously
   inspected by the old notebook and is explicitly **not** a new holdout.
7. Metrics include prevalence, ROC-AUC, average precision and lift, Brier and
   Brier skill, log loss, predicted versus observed incidence, calibration
   intercept/slope, distinct events, and reliability counts/uncertainty.
   Country bootstrap intervals preserve full country histories and are clearly
   fixed-prediction intervals; they do not include refitting uncertainty.

The current small feature set uses available spell duration and lagged V-Dem
polyarchy. WDI ingestion remains supported, but no WDI vintage was preserved
by the original commit, so macro features are not silently fetched or used in
the offline reference model. That limitation is preferable to irreproducible
current-API data.

## Inputs, identity, and joins

The committed ERT file is a 19,678 × 42 snapshot with years 1900–2024 and SHA-256
`39c6ca13363765ec4acf2d4c51052ed531bb49c91937924e0250d5384ad7fa4d`.
Its upstream release/tag was not recorded in the initial commit, so it is
honestly identified by its content checksum (`vendored-ert-sha256-39c6ca133637`)
rather than assigned an unverifiable release. See `data/README.md`.

`country_text_id` is the stable political-panel `country_id` and join key; its
version is `ert-country-text-id-v1`. WDI inputs must use genuine ISO3 keys and
World Bank country metadata to remove aggregates. Unmatched and historical
units must be reported, not fuzzy-matched. No incompatible geography is
silently mapped.

## Layout

- `config/model.json`: frozen estimand, timing, cohorts, promotion criteria.
- `src/features.py`: validation, spells, labels, calendar-safe features.
- `src/model.py`: temporal folds, models, calibration, metrics/bootstrap.
- `src/run_pipeline.py`: offline training, evaluation, and latest-origin scoring.
- `src/artifacts.py`, `schemas/`: artifact loading and shared score contract.
- `src/case_audit.py`, `docs/CASE_AUDIT.md`: positive-origin and distinct-event diagnostics.
- `tests/`: synthetic transition, censoring, leakage, boundary, and round-trip tests.
- `notebooks/`: frozen source input and a thin shared-module walkthrough.

## Remaining limitations

- The input lacks a verifiable upstream release tag and historical publication
  vintages; the checksum makes the run repeatable but not vintage-correct.
- The score feature is from the same current V-Dem release as the outcome and
  may embody retrospective revisions.
- Event counts are small and overlapping three-year labels are dependent.
- Normal-approximation reliability intervals are descriptive; subgroup claims
  are withheld because validated region/income metadata are not frozen.
- Country-bootstrap intervals condition on fitted predictions. Model-fitting,
  input-revision, and measurement uncertainty are not included.
- Latest rows are labeled retrospective/research-only. They are not public
  operational country-risk scores.

Sources: [V-Dem ERT repository](https://github.com/vdeminstitute/ERT),
[V-Dem dataset documentation](https://v-dem.net/data/the-v-dem-dataset/), and
[World Bank Indicators API](https://datahelpdesk.worldbank.org/knowledgebase/articles/889392-about-the-indicators-api-documentation).
