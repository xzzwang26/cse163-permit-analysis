"""
Exploratory analysis of Seattle building permit review durations.

The project asks where delays happen in Seattle's permitting pipeline and
which permit attributes predict them. This script loads three datasets
(building permits, plan review steps, and census tract demographics), cleans
them, and produces the summary tables, figures, and statistical tests that
back the EDA report.
"""

import os
import zipfile

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

DATA = "data"
FIG = "figures"

# Columns pulled from each raw CSV. Reading only these keeps the two large
# Seattle files (100-200 MB each) manageable in memory.
PERMIT_COLS = [
    "PermitNum", "PermitClassMapped", "PermitTypeMapped",
    "EstProjectCost", "TotalDaysPlanReview", "DaysOutCorrections",
    "NumberReviewCycles", "HousingUnitsAdded", "AppliedDate", "IssuedDate",
    "Latitude", "Longitude", "OriginalZip", "StatusCurrent",
]
REVIEW_COLS = [
    "PermitNum", "ReviewCycle", "ReviewType", "ReviewTeam",
    "ReviewResultDesc", "ReviewComplexityDesc", "ReviewerAssignDate",
    "ReviewerFinishDate", "TotalDaysPlanReview", "DaysOutCorrections",
    "NumberReviewCycles", "Latitude", "Longitude",
]


def _to_numeric(series):
    """Turn a column with comma-formatted strings into floats.

    Socrata returns some numeric fields as strings like '4,318', so they have
    to be stripped before conversion.
    """
    return pd.to_numeric(series.astype(str).str.replace(",", ""), errors="coerce")


def _parse_dates(series):
    """Parse either ISO or '2018 Aug 31 12:00:00 AM' style timestamps."""
    return pd.to_datetime(series, errors="coerce", format="mixed")


def load_permits():
    """Load and clean the building permits table (one row per application)."""
    df = pd.read_csv(
        os.path.join(DATA, "building_permits.csv"),
        usecols=PERMIT_COLS, low_memory=False,
    )
    for col in ["EstProjectCost", "TotalDaysPlanReview", "DaysOutCorrections",
                "NumberReviewCycles", "HousingUnitsAdded"]:
        df[col] = _to_numeric(df[col])
    for col in ["AppliedDate", "IssuedDate"]:
        df[col] = _parse_dates(df[col])
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    return df


def load_reviews():
    """Load and clean the plan review table (one row per review step)."""
    # The raw CSV has a few rows with unterminated quotes, so use the Python
    # parser and skip the broken lines rather than fail the whole read.
    df = pd.read_csv(
        os.path.join(DATA, "plan_review.csv"),
        usecols=REVIEW_COLS,
        engine="python", on_bad_lines="skip",
    )
    for col in ["ReviewCycle", "TotalDaysPlanReview", "DaysOutCorrections",
                "NumberReviewCycles"]:
        df[col] = _to_numeric(df[col])
    for col in ["ReviewerAssignDate", "ReviewerFinishDate"]:
        df[col] = _parse_dates(df[col])
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    return df


def load_acs():
    """Load census tract demographics; negatives are census missing codes."""
    df = pd.read_csv(os.path.join(DATA, "acs_tracts.csv"), dtype={"GEOID": str})
    for col in df.columns:
        if col not in ("GEOID", "NAME"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] < 0, col] = np.nan
    return df


def seven_number_summary(series):
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


def missingness(df):
    """Return a per-column missing data table sorted by missingness."""
    table = pd.DataFrame({
        "column": df.columns,
        "missing": df.isna().sum().values,
        "pct": (df.isna().mean() * 100).round(2).values,
    }).sort_values("missing", ascending=False)
    return table


def quantitative_summaries(df, cols):
    """Build a table of seven-number summaries for the given columns."""
    rows = []
    for col in cols:
        row = seven_number_summary(df[col])
        row["variable"] = col
        rows.append(row)
    return pd.DataFrame(rows).set_index("variable").round(2)


def categorical_counts(series, name):
    """Return value counts as a tidy two-column table."""
    out = series.value_counts(dropna=False).rename_axis(name).reset_index(name="count")
    return out


def save_figure(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_duration_histogram(df, col, title, filename):
    """Log-scaled histogram for a right-skewed duration variable."""
    fig, ax = plt.subplots(figsize=(7, 4))
    vals = df[col].dropna()
    ax.hist(vals, bins=60, color="#4C72B0", alpha=0.85)
    ax.set_xlabel("Days")
    ax.set_ylabel("Number of permits")
    ax.set_yscale("log")
    ax.set_title(title)
    save_figure(fig, filename)


def plot_class_boxplot(df, col, group, title, filename):
    """Side-by-side boxplots comparing a duration across two groups."""
    fig, ax = plt.subplots(figsize=(6, 4))
    data = df.dropna(subset=[col, group])
    sns.boxplot(data=data, x=group, y=col, hue=group, legend=False,
                ax=ax, palette="muted")
    ax.set_ylabel("Days")
    ax.set_title(title)
    save_figure(fig, filename)


def plot_category_counts(counts, title, xlabel, filename, n=15):
    """Horizontal bar chart of the most common categories."""
    top = counts.head(n)
    fig, ax = plt.subplots(figsize=(6, max(3, 0.3 * len(top))))
    sns.barplot(y=top.iloc[:, 0], x=top["count"], hue=top.iloc[:, 0],
                legend=False, ax=ax, palette="viridis")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    save_figure(fig, filename)


def plot_income_histogram(acs, filename):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(acs["med_income"].dropna(), bins=40, color="#55A868", alpha=0.85)
    ax.set_xlabel("Median household income ($)")
    ax.set_ylabel("Number of census tracts")
    ax.set_title("Distribution of tract median household income in King County")
    save_figure(fig, filename)


def plot_race_composition(acs, filename):
    """Stacked bars of tract racial composition, sorted by non-white share."""
    race_cols = ["white", "black", "asian", "aian", "two_or_more"]
    comp = acs[race_cols].div(acs[race_cols].sum(axis=1), axis=0)
    comp["non_white"] = 1 - comp["white"]
    comp = comp.sort_values("non_white").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    comp[race_cols].plot(kind="bar", stacked=True, ax=ax, width=1,
                         color=["#C44E52", "#8172B3", "#4C72B0",
                                "#DD8452", "#8C8C8C"])
    ax.set_xlabel("Census tracts (sorted by non-white share)")
    ax.set_ylabel("Share of population")
    ax.set_title("Racial composition across King County tracts")
    ax.set_xticks([])
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    save_figure(fig, filename)


def load_tracts():
    """Read the 2020 tract geometry and keep King County only."""
    zpath = os.path.join(DATA, "tl_2020_53_tract.zip")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(DATA)
    shp = os.path.join(DATA, "tl_2020_53_tract.shp")
    tracts = gpd.read_file(shp)
    return tracts[tracts["COUNTYFP"] == "033"].to_crs(epsg=4326)


def spatial_join_permits(permits, tracts):
    """Attach a census tract GEOID to each permit using its coordinates."""
    located = permits.dropna(subset=["Latitude", "Longitude"]).copy()
    pts = gpd.GeoDataFrame(
        located,
        geometry=gpd.points_from_xy(located["Longitude"], located["Latitude"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(pts, tracts[["GEOID", "geometry"]], how="left",
                       predicate="within")
    return joined


def plot_duration_choropleth(permit_tracts, tracts, filename):
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


def plot_income_vs_duration(permit_tracts, filename):
    """Boxplots of review duration across income quintiles."""
    med = permit_tracts.groupby("GEOID")["TotalDaysPlanReview"].median()
    income = permit_tracts.groupby("GEOID")["med_income"].median()
    frame = pd.DataFrame({"median_days": med, "med_income": income}).dropna()
    frame["quintile"] = pd.qcut(frame["med_income"], 5, labels=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=frame, x="quintile", y="median_days", hue="quintile",
                legend=False, ax=ax, palette="Blues")
    ax.set_xlabel("Tract income quintile (1 = lowest)")
    ax.set_ylabel("Median review days")
    ax.set_title("Review duration by tract income quintile")
    save_figure(fig, filename)


def run_statistical_tests(permits, reviews):
    """Run the hypothesis tests tied to the research questions and print them."""
    results = {}

    # Negative durations are data errors, so drop them before any test.
    dur = permits.dropna(subset=["TotalDaysPlanReview", "PermitClassMapped"])
    dur = dur[dur["TotalDaysPlanReview"] >= 0]

    # RQ1: do residential and non-residential permits differ in duration?
    res = dur[dur["PermitClassMapped"] == "Residential"]["TotalDaysPlanReview"]
    non = dur[dur["PermitClassMapped"] == "Non-Residential"]["TotalDaysPlanReview"]
    u, p = stats.mannwhitneyu(res, non, alternative="two-sided")
    results["mann_whitney"] = {"u": u, "p": p,
                               "res_median": res.median(),
                               "non_median": non.median(),
                               "res_n": len(res), "non_n": len(non)}
    print("Mann-Whitney U (residential vs non-residential duration):")
    print(f"  U={u:.1f} p={p:.3e} residential_median={res.median():.1f} "
          f"(n={len(res)}) non_residential_median={non.median():.1f} "
          f"(n={len(non)})")

    # RQ3a: number of correction cycles vs total duration.
    cyc = dur.dropna(subset=["NumberReviewCycles"])
    rho, p = stats.spearmanr(cyc["NumberReviewCycles"], cyc["TotalDaysPlanReview"])
    results["spearman"] = {"rho": rho, "p": p, "n": len(cyc)}
    print("Spearman (review cycles vs duration):")
    print(f"  rho={rho:.3f} p={p:.3e} n={len(cyc)}")

    # RQ3b: are corrections concentrated in specific review types? Collapse
    # rare types into "Other" and use a binary outcome so the expected cell
    # counts clear the chi-square minimum.
    rev = reviews.copy()
    rev["corrections"] = rev["ReviewResultDesc"] == "Corrections Required"
    top_types = rev["ReviewType"].value_counts().head(15).index
    rev["ReviewTypeTop"] = rev["ReviewType"].where(
        rev["ReviewType"].isin(top_types), "Other")
    tab = pd.crosstab(rev["ReviewTypeTop"], rev["corrections"])
    chi2, p, dof, expected = stats.chi2_contingency(tab)
    results["chi_square"] = {"chi2": chi2, "p": p, "dof": dof,
                             "min_expected": expected.min()}
    print("Chi-square (review type vs corrections outcome):")
    print(f"  chi2={chi2:.1f} p={p:.3e} dof={dof} min_expected={expected.min():.1f}")

    return results


def main():
    os.makedirs(FIG, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    permits = load_permits()
    reviews = load_reviews()
    acs = load_acs()

    print("Shapes:", permits.shape, reviews.shape, acs.shape)

    # Missingness for each raw dataset.
    for name, df in [("permits", permits), ("reviews", reviews),
                     ("acs", acs)]:
        print(f"\n--- missingness: {name} ---")
        print(missingness(df).head(12).to_string(index=False))

    # Quantitative summaries of the variables of interest.
    print("\n--- permit quantitative summaries ---")
    print(quantitative_summaries(
        permits, ["TotalDaysPlanReview", "DaysOutCorrections",
                  "NumberReviewCycles", "EstProjectCost",
                  "HousingUnitsAdded"]).to_string())

    # Categorical counts.
    print("\nnegative durations in permits:",
          (permits["TotalDaysPlanReview"] < 0).sum())
    print("\nPermitClassMapped counts:")
    class_counts = categorical_counts(permits["PermitClassMapped"], "class")
    print(class_counts.to_string(index=False))
    print("\nReviewResultDesc counts:")
    result_counts = categorical_counts(reviews["ReviewResultDesc"], "result")
    print(result_counts.to_string(index=False))

    # Figures for the permits table.
    plot_duration_histogram(
        permits, "TotalDaysPlanReview",
        "Distribution of building permit review durations (log scale)",
        "permits_duration_hist.png")
    plot_class_boxplot(
        permits, "TotalDaysPlanReview", "PermitClassMapped",
        "Review duration by permit class", "permits_class_boxplot.png")
    plot_category_counts(
        categorical_counts(permits["PermitTypeMapped"], "type"),
        "Most common permit types", "Number of permits", "permits_type_counts.png")

    # Figures for the reviews table.
    plot_category_counts(
        categorical_counts(reviews["ReviewResultDesc"], "result"),
        "Review outcomes across all plan review steps", "Number of review steps",
        "reviews_result_counts.png", n=10)
    plot_category_counts(
        categorical_counts(reviews["ReviewType"], "type"),
        "Most common review types", "Number of review steps",
        "reviews_type_counts.png")

    # Figures for the ACS table.
    plot_income_histogram(acs, "acs_income_hist.png")
    plot_race_composition(acs, "acs_race_composition.png")

    # Join 1: permits to reviews (one permit, many review steps).
    permit_keys = ["PermitNum", "PermitClassMapped", "TotalDaysPlanReview",
                   "NumberReviewCycles"]
    review_keys = ["PermitNum", "ReviewType", "ReviewResultDesc",
                   "ReviewCycle", "ReviewerAssignDate", "ReviewerFinishDate"]
    merged_pr = permits[permit_keys].merge(reviews[review_keys],
                                           on="PermitNum", how="inner")
    print(f"\npermits x reviews merged shape: {merged_pr.shape}")
    plot_class_boxplot(
        merged_pr, "ReviewCycle", "ReviewResultDesc",
        "Review cycle count by review outcome", "joined_review_cycle_boxplot.png")

    # Join 2: permits to census tracts by coordinates.
    tracts = load_tracts()
    permit_tracts = spatial_join_permits(permits, tracts)
    acs_join = permit_tracts.merge(acs, on="GEOID", how="left")
    print(f"\npermits x tracts joined shape: {acs_join.shape}")
    print("permits matched to a tract:",
          acs_join["GEOID"].notna().mean().round(3))
    plot_duration_choropleth(acs_join, tracts, "joined_duration_choropleth.png")
    plot_income_vs_duration(acs_join, "joined_income_vs_duration.png")

    # Statistical tests for the result validity goal.
    print("\n--- statistical tests ---")
    run_statistical_tests(permits, reviews)

    print("\nEDA complete. Figures written to", FIG)


if __name__ == "__main__":
    main()
