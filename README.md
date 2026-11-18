# CSE 163 Final Project: Modeling Building Permit Review Duration in Seattle

Exploratory data analysis for CSE 163 (Autumn 2026). The project asks where
delays happen in Seattle's construction permitting pipeline and which permit
attributes predict them.

## Files

- `eda.py` loads, cleans, joins, and summarizes three datasets, generates the
  figures, and runs the hypothesis tests
- `test_eda.py` holds assert-based tests for the cleaning and summary helpers
- `eda_report.md` and `eda.pdf` are the EDA report
- `figures/` contains the plots produced by `eda.py`

## Data

The three datasets are public. The two Seattle tables download as CSV from the
Socrata API, and the census tract demographics come from the Census Data API.

- Building Permits: https://data.seattle.gov/dataset/76t5-zqzr
- Plan Review: https://data.seattle.gov/d/tqk8-y2z5
- ACS 5-Year Estimates: https://api.census.gov/data/2024/acs/acs5

The large CSVs are not committed; download them from the URLs above and place
them in a `data/` directory before running `eda.py`. The Washington tract
shapefile is also fetched at runtime.
