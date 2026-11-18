"""
Mark Wang and William Wang
CSE 163
Exploratory analysis of Seattle building permit review durations.

The project asks where delays happen in Seattle's permitting pipeline and
which permit attributes predict them. This script loads three datasets
(building permits, plan review steps, and census tract demographics), cleans
them, and produces the summary tables, figures, and statistical tests that
back the EDA report. Raw-file checks (missingness, impossible values) are
reported before cleaning; every summary, figure, and test after that point
uses the cleaned data with negative durations removed.
"""

from __future__ import annotations

import os
import zipfile

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from scipy import stats

DATA = "data"
FIG = "figures"

PERMIT_COLS = [
    "PermitNum", "PermitClassMapped", "PermitTypeMapped",
    "EstProjectCost", "TotalDaysPlanReview", "DaysOutCorrections",
    "NumberReviewCycles", "HousingUnitsAdded", "AppliedDate", "IssuedDate",
    "Latitude", "Longitude", "OriginalZip", "StatusCurrent",
    "DaysInitialPlanReview", "DaysPlanReviewCity",
]
REVIEW_COLS = [
    "PermitNum", "ReviewCycle", "ReviewType", "ReviewTeam",
    "ReviewResultDesc", "ReviewComplexityDesc", "ReviewerAssignDate",
    "ReviewerFinishDate", "TotalDaysPlanReview", "DaysOutCorrections",
    "NumberReviewCycles", "Latitude", "Longitude",
]

PERMIT_QUANT = [
    "TotalDaysPlanReview", "DaysInitialPlanReview", "DaysPlanReviewCity",
    "DaysOutCorrections", "NumberReviewCycles", "EstProjectCost",
    "HousingUnitsAdded",
]
REVIEW_QUANT = ["ReviewCycle", "step_days"]
ACS_QUANT = ["med_income", "population"]


def _to_numeric(series: pd.Series) -> pd.Series:
    """Turn a column with comma-formatted strings into floats.

    Socrata returns some numeric fields as strings like '4,318', so they have
    to be stripped before conversion.
    """
    return pd.to_numeric(series.astype(str).str.replace(",", ""),
                         errors="coerce")


def _parse_dates(series: pd.Series) -> pd.Series:
    """Parse either ISO or '2018 Aug 31 12:00:00 AM' style timestamps."""
    return pd.to_datetime(series, errors="coerce", format="mixed")


def load_permits() -> pd.DataFrame:
    """Load and clean formats in the building permits table."""
    df = pd.read_csv(os.path.join(DATA, "building_permits.csv"),
                     usecols=PERMIT_COLS, low_memory=False)
    for col in PERMIT_QUANT:
        df[col] = _to_numeric(df[col])
    for col in ["AppliedDate", "IssuedDate"]:
        df[col] = _parse_dates(df[col])
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    return df


def load_reviews() -> pd.DataFrame:
    """Load and clean formats in the plan review table.

    The raw CSV has a few rows with unterminated quotes, so use the Python
    parser and skip the broken lines rather than fail the whole read.
    """
    df = pd.read_csv(os.path.join(DATA, "plan_review.csv"),
                     usecols=REVIEW_COLS,
                     engine="python", on_bad_lines="skip")
    for col in ["ReviewCycle", "TotalDaysPlanReview", "DaysOutCorrections",
                "NumberReviewCycles"]:
        df[col] = _to_numeric(df[col])
    for col in ["ReviewerAssignDate", "ReviewerFinishDate"]:
        df[col] = _parse_dates(df[col])
    df["step_days"] = ((df["ReviewerFinishDate"]
                        - df["ReviewerAssignDate"]).dt.days)
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    return df


def load_acs() -> pd.DataFrame:
    """Load census tract demographics; negatives are census missing codes."""
    df = pd.read_csv(os.path.join(DATA, "acs_tracts.csv"),
                     dtype={"GEOID": str})
    for col in df.columns:
        if col not in ("GEOID", "NAME"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] < 0, col] = np.nan
    return df


def seven_number_summary(series: pd.Series) -> dict:
    """Return mean, std, min, Q1, median, Q3, max as a single row."""
    s = series.dropna()
    return {
        "mean": s.mean(),
        "std": s.std(),
        "min": s.min(),
        "q1": s.quantile(0.25),
        "median": s.median(),
        "q3": s.quantile(0.75),
        "max": s.max(),
    }


def missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-column missing data table sorted by missingness."""
    return pd.DataFrame({
        "column": df.columns,
        "missing": df.isna().sum().values,
        "pct": (df.isna().mean() * 100).round(2).values,
    }).sort_values("missing", ascending=False)


def quantitative_summaries(df: pd.DataFrame,
                           cols: list[str]) -> pd.DataFrame:
    """Build a table of seven-number summaries for the given columns."""
    rows = []
    for col in cols:
        row = seven_number_summary(df[col])
        row["variable"] = col
        rows.append(row)
    return pd.DataFrame(rows).set_index("variable").round(2)


def categorical_counts(series: pd.Series, name: str) -> pd.DataFrame:
    """Return value counts as a tidy two-column table."""
    out = series.value_counts(dropna=False).rename_axis(name)
    return out.reset_index(name="count")


def save_figure(fig: Figure, name: str) -> None:
    """Write a figure to the figures directory and close it."""
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_duration_histogram(df: pd.DataFrame, col: str, title: str,
                            filename: str) -> None:
    """Log-scaled histogram for a right-skewed duration variable."""
    fig, ax = plt.subplots(figsize=(7, 4))
    vals = df[col].dropna()
    ax.hist(vals, bins=60, color="#4C72B0", alpha=0.85)
    ax.set_xlabel("Days")
    ax.set_ylabel("Number of permits")
    ax.set_yscale("log")
    ax.set_title(title)
    save_figure(fig, filename)


def plot_group_boxplot(df: pd.DataFrame, col: str, group: str, title: str,
                       filename: str, ylabel: str = "Days") -> None:
    """Boxplots comparing one variable across the groups of another."""
    fig, ax = plt.subplots(figsize=(6, 4))
    data = df.dropna(subset=[col, group])
    sns.boxplot(data=data, x=group, y=col, hue=group, legend=False,
                ax=ax, palette="muted")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    save_figure(fig, filename)


def plot_category_counts(counts: pd.DataFrame, title: str, xlabel: str,
                         filename: str, n: int = 15) -> None:
    """Horizontal bar chart of the most common categories."""
    top = counts.head(n)
    fig, ax = plt.subplots(figsize=(6, max(3, 0.3 * len(top))))
    sns.barplot(y=top.iloc[:, 0], x=top["count"], hue=top.iloc[:, 0],
                legend=False, ax=ax, palette="viridis")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    save_figure(fig, filename)


def plot_income_histogram(acs: pd.DataFrame, filename: str) -> None:
    """Histogram of tract median household income."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(acs["med_income"].dropna(), bins=40, color="#55A868", alpha=0.85)
    ax.set_xlabel("Median household income ($)")
    ax.set_ylabel("Number of census tracts")
    ax.set_title("Tract median household income in King County")
    save_figure(fig, filename)


def plot_race_composition(acs: pd.DataFrame, filename: str) -> None:
    """Racial composition of tracts, grouped by income quintile.

    One bar per tract is unreadable at 495 tracts, so the composition is
    aggregated to five income quintiles instead.
    """
    race_cols = ["white", "black", "asian", "aian", "two_or_more"]
    comp = acs.dropna(subset=["med_income"]).copy()
    comp["quintile"] = pd.qcut(comp["med_income"], 5, labels=False)
    shares = comp.groupby("quintile")[race_cols].sum()
    shares = shares.div(shares.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    shares.plot(kind="bar", stacked=True, ax=ax, width=0.8,
                color=["#4C72B0", "#8172B3", "#C44E52", "#DD8452", "#8C8C8C"])
    ax.set_xticklabels(["1 (lowest)", "2", "3", "4", "5 (highest)"],
                       rotation=0)
    ax.set_xlabel("Tract income quintile")
    ax.set_ylabel("Share of population")
    ax.set_title("Racial composition by tract income quintile")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    save_figure(fig, filename)


def plot_cycle_correction_rate(reviews: pd.DataFrame,
                               filename: str) -> None:
    """Share of review steps ending in corrections, by review cycle."""
    rev = reviews.dropna(subset=["ReviewCycle"]).copy()
    rev["corrections"] = rev["ReviewResultDesc"] == "Corrections Required"
    g = rev.groupby("ReviewCycle")["corrections"].agg(["mean", "size"])
    g = g[g["size"] >= 50].head(8)
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = g.index.astype(int)
    sns.barplot(x=labels, y=g["mean"], hue=labels, legend=False, ax=ax,
                palette="flare")
    for i, (rate, count) in enumerate(zip(g["mean"].values,
                                          g["size"].values)):
        ax.text(i, rate + 0.015, f"n={int(count):,}", ha="center",
                fontsize=8)
    ax.set_xlabel("Review cycle")
    ax.set_ylabel("Share of steps requiring corrections")
    ax.set_ylim(0, 0.72)
    ax.set_title("Correction rate declines over review cycles")
    save_figure(fig, filename)


def load_tracts() -> gpd.GeoDataFrame:
    """Read the 2020 tract geometry and keep King County only."""
    zpath = os.path.join(DATA, "tl_2020_53_tract.zip")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(DATA)
    shp = os.path.join(DATA, "tl_2020_53_tract.shp")
    tracts = gpd.read_file(shp)
    return tracts[tracts["COUNTYFP"] == "033"].to_crs(epsg=4326)


def spatial_join_permits(permits: pd.DataFrame,
                         tracts: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Attach a census tract GEOID to each permit using its coordinates."""
    located = permits.dropna(subset=["Latitude", "Longitude"]).copy()
    pts = gpd.GeoDataFrame(
        located,
        geometry=gpd.points_from_xy(located["Longitude"], located["Latitude"]),
        crs="EPSG:4326",
    )
    return gpd.sjoin(pts, tracts[["GEOID", "geometry"]], how="left",
                     predicate="within")


def plot_duration_choropleth(permit_tracts: gpd.GeoDataFrame,
                             tracts: gpd.GeoDataFrame,
                             filename: str) -> None:
    """Map the median plan review duration by census tract."""
    med = permit_tracts.groupby("GEOID")["TotalDaysPlanReview"].median()
    merged = tracts[["GEOID", "geometry"]].merge(med, on="GEOID", how="left")
    fig, ax = plt.subplots(figsize=(9, 8))
    merged.plot(column="TotalDaysPlanReview", ax=ax, cmap="YlOrRd",
                legend=True, legend_kwds={"label": "Median days"},
                missing_kwds={"color": "#eeeeee"})
    ax.set_axis_off()
    ax.set_title("Median permit review duration by census tract")
    save_figure(fig, filename)


def plot_income_vs_duration(permit_tracts: gpd.GeoDataFrame,
                            filename: str) -> None:
    """Boxplots of review duration across income quintiles."""
    med = permit_tracts.groupby("GEOID")["TotalDaysPlanReview"].median()
    income = permit_tracts.groupby("GEOID")["med_income"].median()
    frame = pd.DataFrame({"median_days": med, "med_income": income}).dropna()
    frame["quintile"] = pd.qcut(frame["med_income"], 5,
                                labels=[1, 2, 3, 4, 5])
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=frame, x="quintile", y="median_days", hue="quintile",
                legend=False, ax=ax, palette="Blues")
    ax.set_xlabel("Tract income quintile (1 = lowest)")
    ax.set_ylabel("Median review days")
    ax.set_title("Review duration by tract income quintile")
    save_figure(fig, filename)


def print_missingness_deep_dive(permits: pd.DataFrame) -> None:
    """Compare permits with and without a duration on status, year, type.

    The 82.6% missingness claim in the report needs evidence, so this breaks
    the missing rate down along the three axes most likely to explain it.
    """
    p = permits.copy()
    p["has_dur"] = p["TotalDaysPlanReview"].notna()
    p["app_year"] = p["AppliedDate"].dt.year

    print("\n--- duration missingness by StatusCurrent ---")
    print(p.groupby("StatusCurrent")["has_dur"].agg(["mean", "size"])
           .sort_values("size", ascending=False).round(3).to_string())
    print("\n--- duration missingness by application year ---")
    print(p.groupby("app_year")["has_dur"].agg(["mean", "size"])
           .round(3).to_string())
    print("\n--- duration missingness by PermitTypeMapped (n>=500) ---")
    t = p.groupby("PermitTypeMapped")["has_dur"].agg(["mean", "size"])
    print(t[t["size"] >= 500].sort_values("mean").round(3).to_string())


def run_statistical_tests(permits: pd.DataFrame,
                          reviews: pd.DataFrame) -> dict:
    """Run the hypothesis tests tied to the research questions."""
    results = {}

    # Negative durations are data errors, so drop them before any test.
    dur = permits.dropna(subset=["TotalDaysPlanReview", "PermitClassMapped"])
    dur = dur[dur["TotalDaysPlanReview"] >= 0]

    # RQ1: do duration distributions differ by permit class? MWU tests the
    # whole distribution, so medians are reported alongside as description.
    res = dur[dur["PermitClassMapped"] == "Residential"]["TotalDaysPlanReview"]
    non = dur[dur["PermitClassMapped"] == "Non-Residential"][
        "TotalDaysPlanReview"]
    u, p = stats.mannwhitneyu(res, non, alternative="two-sided")
    results["mann_whitney"] = {"u": u, "p": p,
                               "res_median": res.median(),
                               "non_median": non.median(),
                               "res_n": len(res), "non_n": len(non)}
    print("Mann-Whitney U (residential vs non-residential duration):")
    print(f"  U={u:.3g} p={p:.3e} residential_median={res.median():.1f} "
          f"(n={len(res)}) non_residential_median={non.median():.1f} "
          f"(n={len(non)})")

    # RQ3a: review cycles vs duration. Cycle count is a process variable
    # unknown at application time, so this supports the story but is not a
    # predictive feature (see the ML note in the report).
    cyc = dur.dropna(subset=["NumberReviewCycles"])
    rho, p = stats.spearmanr(cyc["NumberReviewCycles"],
                             cyc["TotalDaysPlanReview"])
    results["spearman"] = {"rho": rho, "p": p, "n": len(cyc)}
    print("Spearman (review cycles vs duration):")
    print(f"  rho={rho:.3f} p={p:.3e} n={len(cyc)}")

    # RQ3 direct: how much slower are permits that went through corrections?
    revised = dur[dur["NumberReviewCycles"] >= 2]["TotalDaysPlanReview"]
    first = dur[dur["NumberReviewCycles"] <= 1]["TotalDaysPlanReview"]
    u2, p2 = stats.mannwhitneyu(revised, first, alternative="two-sided")
    results["rq3_duration"] = {"u": u2, "p": p2,
                               "revised_median": revised.median(),
                               "first_median": first.median(),
                               "revised_n": len(revised),
                               "first_n": len(first)}
    print("Mann-Whitney U (revised vs first-pass duration):")
    print(f"  U={u2:.3g} p={p2:.3e} revised_median={revised.median():.1f} "
          f"(n={len(revised)}) first_pass_median={first.median():.1f} "
          f"(n={len(first)})")

    # RQ3b: are corrections concentrated in specific review types? The unit
    # is the first result of each permit-review-type pair, which avoids
    # counting every correction cycle of the same permit as separate
    # evidence. Residual dependence across types within one permit remains
    # and is stated as a limitation in the report.
    rev = reviews.dropna(subset=["ReviewCycle"]).sort_values("ReviewCycle")
    rev = rev.drop_duplicates(subset=["PermitNum", "ReviewType"],
                              keep="first")
    rev["corrections"] = rev["ReviewResultDesc"] == "Corrections Required"
    top = rev["ReviewType"].value_counts().head(15).index
    rev["type_top"] = rev["ReviewType"].where(rev["ReviewType"].isin(top),
                                              "Other")
    tab = pd.crosstab(rev["type_top"], rev["corrections"])
    chi2, p, dof, expected = stats.chi2_contingency(tab)
    rates = (tab[True] / (tab[True] + tab[False])).sort_values(
        ascending=False)
    results["chi_square"] = {"chi2": chi2, "p": p, "dof": dof,
                             "min_expected": expected.min(),
                             "n_units": len(tab)}
    results["chi_rates"] = rates
    print("Chi-square (review type vs first-pass corrections outcome):")
    print(f"  units={len(tab)} chi2={chi2:.1f} p={p:.3e} dof={dof} "
          f"min_expected={expected.min():.1f}")
    print("  highest correction rates:")
    print(rates.head(5).round(3).to_string())
    print("  lowest correction rates:")
    print(rates.tail(5).round(3).to_string())
    return results


def main():
    os.makedirs(FIG, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    permits = load_permits()
    reviews = load_reviews()
    acs = load_acs()
    print("Shapes:", permits.shape, reviews.shape, acs.shape)

    # ---- raw-file checks, before any cleaning ----
    for name, df in [("permits", permits), ("reviews", reviews),
                     ("acs", acs)]:
        print(f"\n--- missingness: {name} ---")
        print(missingness(df).head(12).to_string(index=False))

    neg = permits[permits["TotalDaysPlanReview"] < 0]
    print(f"\nnegative durations in permits: {len(neg)}")
    print(neg["StatusCurrent"].value_counts().to_string())

    print("\n--- raw (uncleaned) duration summary ---")
    print(quantitative_summaries(permits, ["TotalDaysPlanReview"])
          .to_string())

    # ---- cleaning ----
    permits_clean = permits[permits["TotalDaysPlanReview"] >= 0].copy()

    # ---- missingness deep dive on the raw file ----
    print_missingness_deep_dive(permits)

    # ---- summaries on cleaned data ----
    print("\n--- cleaned permit quantitative summaries ---")
    print(quantitative_summaries(permits_clean, PERMIT_QUANT).to_string())
    print("\n--- review quantitative summaries ---")
    print(quantitative_summaries(reviews, REVIEW_QUANT).to_string())
    print("\n--- step days by review type (n>=200) ---")
    sd = reviews.dropna(subset=["step_days"])
    sd = sd[sd["step_days"] >= 0]
    g2 = sd.groupby("ReviewType")["step_days"].agg(["median", "mean", "size"])
    print(g2[g2["size"] >= 200].sort_values("median", ascending=False)
          .head(12).round(1).to_string())
    print("\n--- ACS quantitative summaries ---")
    print(quantitative_summaries(acs, ACS_QUANT).to_string())

    print("\nPermitClassMapped counts:")
    class_counts = categorical_counts(permits_clean["PermitClassMapped"],
                                      "class")
    print(class_counts.to_string(index=False))
    print("\nReviewResultDesc counts:")
    result_counts = categorical_counts(reviews["ReviewResultDesc"], "result")
    print(result_counts.to_string(index=False))

    # ---- figures for the permits table (cleaned) ----
    plot_duration_histogram(
        permits_clean, "TotalDaysPlanReview",
        "Distribution of building permit review durations (log scale)",
        "permits_duration_hist.png")
    plot_group_boxplot(
        permits_clean, "TotalDaysPlanReview", "PermitClassMapped",
        "Review duration by permit class", "permits_class_boxplot.png")
    plot_category_counts(
        categorical_counts(permits_clean["PermitTypeMapped"], "type"),
        "Most common permit types", "Number of permits",
        "permits_type_counts.png")

    # ---- figures for the reviews table ----
    plot_category_counts(
        categorical_counts(reviews["ReviewResultDesc"], "result"),
        "Review outcomes across all plan review steps",
        "Number of review steps", "reviews_result_counts.png", n=10)
    plot_category_counts(
        categorical_counts(reviews["ReviewType"], "type"),
        "Most common review types", "Number of review steps",
        "reviews_type_counts.png")

    # ---- figures for the ACS table ----
    plot_income_histogram(acs, "acs_income_hist.png")
    plot_race_composition(acs, "acs_race_composition.png")

    # ---- join 1: permits to reviews ----
    permit_keys = ["PermitNum", "PermitClassMapped", "TotalDaysPlanReview",
                   "NumberReviewCycles"]
    review_keys = ["PermitNum", "ReviewCycle", "ReviewType",
                   "ReviewResultDesc", "ReviewerAssignDate",
                   "ReviewerFinishDate"]
    merged_pr = permits_clean[permit_keys].merge(reviews[review_keys],
                                                 on="PermitNum",
                                                 how="inner")
    steps = merged_pr.groupby("PermitNum").size()
    print(f"\npermits x reviews merged shape: {merged_pr.shape}")
    print(f"review steps per permit: median={steps.median():.0f} "
          f"p90={steps.quantile(0.9):.0f} max={steps.max():.0f}")
    plot_cycle_correction_rate(merged_pr, "joined_cycle_correction_rate.png")

    # ---- join 2: permits to census tracts ----
    tracts = load_tracts()
    permit_tracts = spatial_join_permits(permits_clean, tracts)
    acs_join = permit_tracts.merge(acs, on="GEOID", how="left")
    print(f"\npermits x tracts joined shape: {acs_join.shape}")
    print("permits matched to a tract:",
          acs_join["GEOID"].notna().mean().round(3))
    plot_duration_choropleth(acs_join, tracts,
                             "joined_duration_choropleth.png")
    plot_income_vs_duration(acs_join, "joined_income_vs_duration.png")

    # ---- statistical tests ----
    print("\n--- statistical tests ---")
    run_statistical_tests(permits_clean, reviews)

    print("\nEDA complete. Figures written to", FIG)


if __name__ == "__main__":
    main()
