# Case audit guide

The model target is mechanically derived from the frozen annual V-Dem
classification. A row is not evidence that the project has independently
validated the historical interpretation of a country case. Case review is a
required measurement step before adding model complexity.

Each pipeline run now writes:

- `case_audit/positive_origins.csv`: every positive country-origin label,
  including years until transition, spell context, maturity, and the number of
  origin rows pointing to the same event.
- `case_audit/distinct_events.csv`: one row per country/event year with the
  before/after regime categories, polyarchy change, first subsequent return to
  democracy, boundary flag, and repeated-event flag.
- `case_audit/case_audit_summary.json`: dependence and review counts.

Generate these tables offline with:

```bash
python src/run_pipeline.py --input notebooks/data/ert.csv --output-dir artifacts
```

## Examples in the frozen snapshot

These are examples of what the **dataset encodes**, not independent claims
about the countries:

- Fiji changes from category 2 in 1999 to category 0 in 2000. The three-year
  target therefore marks origins 1997, 1998, and 1999 positive. The first later
  category 2/3 year is 2003, so this is a temporary exit that still counts.
- Hungary changes from category 2 in 2017 to category 1 in 2018. Origins
  2015–2017 are three overlapping positive labels for one distinct event.
- Tunisia changes from category 2 in 2021 to category 1 in 2022. Origins
  2019–2021 point to that event.
- Bangladesh has separate transitions in 1996 and 2002, illustrating why a
  country can contribute events in multiple democratic spells.
- Georgia, Indonesia, and Mongolia change category in 2024. Those events sit at
  the data boundary; only the 2021 origins are administratively mature for a
  three-year binary evaluation as of 2024.

## How the cases suggest improvements

1. **Report events and labels separately.** Three-year windows usually create
   three correlated positive rows for one event. Row-level sample size must not
   be presented as the number of breakdowns, and resampling must preserve
   country histories.
   The audit also flags event windows cut by the 1995 analysis boundary; these
   should not be mistaken for events that intrinsically generate fewer labels.
2. **Review temporary exits explicitly.** They belong in the frozen estimand,
   but a sensitivity analysis can report sustained exits separately rather
   than silently changing the primary target.
3. **Review category-boundary cases.** Most transitions in this snapshot are
   category 2 to category 1. Compare the annual classification with uncertainty
   bounds and release documentation; do not substitute a polyarchy threshold
   after observing model results.
4. **Separate boundary events from mature evaluation.** A known event can be a
   positive label while its origin cohort remains administratively immature.
   This prevents recent positives from entering evaluation when corresponding
   non-events are censored.
5. **Stratify errors by case characteristic.** Join evaluation predictions to
   `distinct_events.csv` and inspect missed events by spell age, temporary exit,
   repeated transition, feature completeness, and calendar origin. Avoid
   unsupported region/income comparisons until frozen metadata are available.
6. **Adjudicate suspicious cases outside the modeling loop.** Create a versioned
   review table with reviewer, source, decision, and rationale. Do not delete a
   difficult positive because the model misses it; any alternative outcome
   definition should be preregistered and evaluated as a sensitivity analysis.
