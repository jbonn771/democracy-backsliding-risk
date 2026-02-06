# Data Sources

## V-Dem Episodes of Regime Transformation (ERT)
The ERT dataset provides annual regime trajectories and identifies episodes of
regime transformation (including autocratization and democratization). This
project uses ERT to define democratic spells and autocratization onsets.

Key fields used:
- `year`
- `v2x_regime` (regime category)
- `country_text_id` / ISO3-like country identifiers (standardized to `country_key`)

The ingestion logic in `src/data.py` standardizes these columns and filters to
years with reliable WDI coverage.

## World Bank Indicators (WDI)
The model uses a compact set of macroeconomic and demographic indicators:

**Core macro**
- GDP growth (`NY.GDP.MKTP.KD.ZG`)
- CPI inflation (`FP.CPI.TOTL.ZG`)
- Unemployment (`SL.UEM.TOTL.ZS`)
- GDP per capita, current USD (`NY.GDP.PCAP.CD`)

**Inequality (sparse)**
- Gini (`SI.POV.GINI`)

**Debt stress**
- External debt to GNI (`DT.DOD.DECT.GN.ZS`)
- Debt service to exports (`DT.TDS.DECT.EX.ZS`)

**Living standards proxy**
- Consumption per capita growth (`NE.CON.PRVT.PC.KD.ZG`)

**Structural / sociological proxies**
- Resource rents (% GDP) (`NY.GDP.TOTL.RT.ZS`)
- Total population (`SP.POP.TOTL`)
- Urban population (`SP.URB.TOTL`)
- Youth population components (`SP.POP.1519.MA`, `SP.POP.1519.FE`, `SP.POP.2024.MA`, `SP.POP.2024.FE`)
- Youth unemployment (`SL.UEM.1524.ZS`)
- Net migration (`SM.POP.NETM`)

Derived features computed in `src/data.py`:
- Youth share (15-24) as percent of total population
- Urban share as percent of total population
- Net migration per 1,000 population
- Inflation surprise (current CPI inflation minus trailing 5-year mean of prior years)

## Why Raw Data Files Aren't Committed
- **Licensing and attribution**: data usage terms require appropriate citation.
- **Reproducibility**: datasets can be re-fetched from authoritative sources.
- **Size and cleanliness**: raw files are large and often revised.

## Citation Guidance
When using this repository or derived results, cite:
- **V-Dem Institute** - Episodes of Regime Transformation (ERT). Include the
  dataset name, year/version, and access date.
- **World Bank** - World Development Indicators (WDI). Include indicator list
  and access date.

Use the official citation formats provided by each data source.
