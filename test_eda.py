"""
Mark Wang and William Wang
CSE 163
Tests for the cleaning and summary helpers in eda.py.

Each test feeds a small hand-built input to a function and asserts the exact
output, so the tests stay fast and don't depend on the full datasets.
"""

from __future__ import annotations

import pandas as pd

import eda


def test_to_numeric() -> None:
    """Tests _to_numeric strips commas and turns junk into NaN."""
    out = eda._to_numeric(pd.Series(["4,318", "709", None, "abc"]))
    assert out.iloc[0] == 4318.0
    assert out.iloc[1] == 709.0
    assert pd.isna(out.iloc[2])
    assert pd.isna(out.iloc[3])


def test_parse_dates() -> None:
    """Tests _parse_dates handles both Socrata timestamp styles."""
    out = eda._parse_dates(
        pd.Series(["2018 Aug 31 12:00:00 AM", "2006-09-29", "not a date"]))
    assert out.iloc[0].year == 2018
    assert out.iloc[1].year == 2006
    assert pd.isna(out.iloc[2])


def test_seven_number_summary() -> None:
    """Tests seven_number_summary on a known vector."""
    result = eda.seven_number_summary(pd.Series([1, 2, 3, 4, 5]))
    assert result["mean"] == 3.0
    assert result["median"] == 3.0
    assert result["min"] == 1.0
    assert result["max"] == 5.0
    assert abs(result["q1"] - 2.0) < 1e-9
    assert abs(result["q3"] - 4.0) < 1e-9


def test_missingness() -> None:
    """Tests missingness computes per-column counts and percentages."""
    frame = pd.DataFrame({"a": [1, None, 3], "b": [None, None, None]})
    table = eda.missingness(frame)
    row_a = table[table["column"] == "a"]
    row_b = table[table["column"] == "b"]
    assert row_a["missing"].iloc[0] == 1
    assert row_b["missing"].iloc[0] == 3
    assert abs(row_a["pct"].iloc[0] - 33.33) < 0.01


def test_categorical_counts() -> None:
    """Tests categorical_counts returns a two-column table with a total."""
    out = eda.categorical_counts(pd.Series(["x", "x", "y"]), "label")
    assert out["count"].sum() == 3
    assert set(out["label"]) == {"x", "y"}


def main():
    test_to_numeric()
    test_parse_dates()
    test_seven_number_summary()
    test_missingness()
    test_categorical_counts()
    print("All tests passed.")


if __name__ == "__main__":
    main()
