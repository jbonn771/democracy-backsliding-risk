# Input provenance and acquisition policy

## Vendored ERT snapshot

`notebooks/data/ert.csv` was already committed in the initial repository. It
contains 19,678 rows, 42 columns, 183 `country_text_id` values, and years
1900–2024. Its SHA-256 is
`39c6ca13363765ec4acf2d4c51052ed531bb49c91937924e0250d5384ad7fa4d`.
The file has an export index column (`Unnamed: 0`), ERT episode fields, Regimes
of the World `v2x_regime`, and `v2x_polyarchy` with uncertainty bounds.

The original commit pointed at the moving ERT `master` CSV but did not record a
release tag, retrieval date, license copy, or checksum. Therefore the frozen
identifier is `vendored-ert-sha256-39c6ca133637`; claiming a named upstream
release would be unsupported. The matching conceptual documentation is the
[V-Dem dataset page](https://v-dem.net/data/the-v-dem-dataset/) and the source
is the [ERT repository](https://github.com/vdeminstitute/ERT). A replacement
snapshot must use a versioned URL, record retrieval UTC, preserve upstream
documentation, and update its manifest and checksum.

## WDI

The initial repository did **not** preserve a WDI snapshot or metadata. The
reference offline model consequently does not fetch a mutable API during
training. `src/data.py` retains validated API acquisition helpers for a future
explicit snapshot. A WDI snapshot manifest must include API URLs, UTC retrieval
time, response schema, indicator code/name/definition/source-note, checksum,
and World Bank country metadata. Aggregates (metadata `region.id == "NA"`) must
be excluded before joining, and unmatched IDs must be emitted as a report.

The intended indicators and codes remain in `DEFAULT_WB_INDICATORS`. The API
documentation is [World Bank Indicators API](https://datahelpdesk.worldbank.org/knowledgebase/articles/889392-about-the-indicators-api-documentation).

## Availability and crosswalk rules

- `country_text_id` is retained as the versioned ERT political identifier
  (`ert-country-text-id-v1`). It is not asserted to be ISO3 for every historical
  entity.
- WDI joins require exact, historically valid crosswalk entries. No name-based
  or fuzzy mapping is allowed.
- Observation year, assumed availability lag, value age, missingness, and
  carry-forward status are distinct. Lagging never proves publication.
- Raw downloads must use the atomic `.partial` then rename implementation and
  pass HTTP, nonempty-body, schema, uniqueness, and checksum checks.
