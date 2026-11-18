# Exploratory Data Analysis: Modeling Building Permit Review Duration in Seattle

Mark Wang and William Wang
CSE 163, Autumn 2026
Part 2: Exploratory Data Analysis
GitHub repository: [PASTE YOUR PUBLIC REPO LINK HERE]

## Summary of Research Questions

1. How is a permit's total time from application to issuance distributed across the stages of plan review, and how does that distribution shift with permit type, project complexity, and neighborhood?
2. Which attributes of a permit, including its estimated cost, housing units, complexity class, and review history, most strongly predict how long review will take, and how accurately can we predict that duration for permits the model has not seen?
3. How much slower are permits that get sent back for corrections than permits approved on the first pass, and are those correction cycles concentrated in particular review types or teams?

## Motivation

Seattle's construction permitting process has a reputation for being slow, and the delay feeds into the cost of housing and how quickly small businesses can open. In 2025 the city became one of the few large U.S. cities to publish machine-readable data covering its entire permitting pipeline, down to individual review steps, so the delay can now be measured and explained instead of only complained about. Knowing which stage is slow and which factors predict it helps applicants set expectations and helps the city decide where to focus. A descriptive study would show that permits are slow; an explanatory model shows why, separating 'this project is inherently complex' from 'this stage is a bottleneck'. The problem sits where urban policy meets applied machine learning, and because every dataset is public and recent, the work can be reproduced outside the course.

## Dataset

We work with the same three public datasets proposed in Part 1, none of which appeared in a CSE 163 assignment or demo. The two Seattle tables were downloaded as CSV through the Socrata API, and the census tables were pulled from the Census Data API. All cleaning, joining, and analysis is done programmatically in Python.

| Dataset | Raw shape | Rows represent | Columns used |
|---|---|---|---|
| Building Permits (data.seattle.gov/dataset/76t5-zqzr) | 194,311 x 41 | One building permit application | PermitNum, PermitClassMapped, PermitTypeMapped, EstProjectCost, TotalDaysPlanReview, DaysOutCorrections, NumberReviewCycles, HousingUnitsAdded, AppliedDate, IssuedDate, Latitude, Longitude, OriginalZip, StatusCurrent |
| Plan Review (data.seattle.gov/d/tqk8-y2z5) | 271,104 x 46 | One review step within one permit | PermitNum, ReviewCycle, ReviewType, ReviewTeam, ReviewResultDesc, ReviewComplexityDesc, ReviewerAssignDate, ReviewerFinishDate, plus the shared duration fields |
| ACS 5-Year Estimates (api.census.gov, 2024 acs5) | 495 x 9 | One census tract in King County | GEOID, median household income (B19013), population (B01003), five race count columns |

During the EDA we found that the raw files are messier than the proposal assumed, which reinforced the Messy Data goal. The numeric duration fields arrive as comma-formatted strings ('4,318'), the two Seattle tables use different date formats ('2006-09-29' versus '2018 Aug 31 12:00:00 AM'), the Plan Review CSV contains a handful of rows with unterminated quotes that break a strict parser, and TotalDaysPlanReview holds 83 negative values that are clearly data errors. All of these are handled in code rather than by hand.

## Method

The EDA proceeded in five steps. First, we loaded only the columns each research question needs, which keeps the two large Seattle files manageable in memory. Second, we cleaned each table: comma-formatted numerics were stripped and converted, both timestamp styles were parsed, and the census missing-value codes (negative counts) were set to NaN. Third, we produced the required summaries for every dataset: row and column counts, a missingness table, seven-number summaries for quantitative variables, and value counts for categorical ones. Fourth, we built the two joins our Multiple Datasets goal requires, permits to reviews on PermitNum and permits to census tracts through a spatial join on permit coordinates, and repeated the key summaries and visualizations on each joined table. Fifth, we ran the three hypothesis tests attached to our Result Validity goal, verifying each test's assumptions on the actual data.

## EDA Results

### Building Permits

The permits table has 194,311 rows and 41 raw columns, of which we use 14. Each row is one application, and the columns describe the project (class, type, cost, housing units), its timeline (applied, issued), its review history (total days, correction days, cycle count), and its location (latitude, longitude, ZIP).

Missing data is substantial and structured. TotalDaysPlanReview is absent for 82.6% of rows, which reflects that most applications in the file were still under review, withdrawn, or of a type that skips plan review. EstProjectCost is missing 18.1%, AppliedDate 23.8%, and IssuedDate 27.3%, while Latitude and Longitude are almost always present (0.6% missing). We also found 83 negative durations, which are impossible values and are excluded from the tests below.

| TotalDaysPlanReview | value |
|---|---|
| mean | 170.12 |
| std | 233.70 |
| min | -743.00 |
| q1 | 44.00 |
| median | 96.00 |
| q3 | 206.00 |
| max | 4,734.00 |

Figure 1: Histogram of permit review durations on a log-scaled count axis. The distribution is heavily right-skewed, with a mode under 100 days and a long tail past 1,000.

[Figure: figures/permits_duration_hist.png]

PermitClassMapped splits cleanly into 135,079 Residential and 52,286 Non-Residential applications, with 6,946 blank. The categorical counts for PermitTypeMapped show that Building, Construction/Land Use, and similar high-level types dominate, while dozens of rare types appear only a handful of times.

Figure 2: Review duration by permit class. Residential permits run visibly longer than non-residential ones, a pattern the Mann-Whitney test confirms below.

[Figure: figures/permits_class_boxplot.png]

Figure 3: The most common permit types by application count.

[Figure: figures/permits_type_counts.png]

### Plan Review

The reviews table has 271,104 rows and 46 raw columns, of which we use 13. Each row is one review step performed by one team on one permit, so a permit that cycles through corrections several times contributes many rows.

Missingness here is far lighter than in the permits table: TotalDaysPlanReview is missing only 8.15% and the reviewer dates under 8%, because this table is written as each step completes rather than backfilled at the permit level.

ReviewResultDesc has four values, and their sizes already hint at the answer to our third research question: Corrections Required appears 139,571 times against 128,920 Approvals, meaning more than half of all review steps send the applicant back.

Figure 4: Review outcomes across all plan review steps.

[Figure: figures/reviews_result_counts.png]

Figure 5: The most common review types by step count. Addressing, Structural, and similar specialist reviews dominate.

[Figure: figures/reviews_type_counts.png]

### ACS Tracts

The census table has 495 rows, one per King County tract, with median household income, population, and five race counts. Only two tracts have a missing income value (0.4%), which we set to NaN after replacing the census missing codes.

Figure 6: Distribution of tract median household income. Tracts range from under 30,000 dollars to over 170,000.

[Figure: figures/acs_income_hist.png]

Figure 7: Racial composition across tracts, sorted by non-white share. The wide spread motivates using tract demographics as a neighborhood control.

[Figure: figures/acs_race_composition.png]

### Join 1: Permits to Reviews

Because one permit goes through many review steps, the inner join on PermitNum keeps all 271,104 review rows and attaches the permit-level class and duration to each step. Every review row found a matching permit, so the join loses nothing.

Figure 8: Review cycle counts by review outcome in the joined table. Steps that end in Corrections Required come disproportionately from later cycles.

[Figure: figures/joined_review_cycle_boxplot.png]

### Join 2: Permits to Census Tracts

We joined permits to the 2020 tract geometry by point-in-polygon on the permit coordinates. 193,174 permits (100% of those with coordinates) landed inside a King County tract, and the ACS table then attaches income and demographics to every one of them.

Figure 9: Median review duration by tract. Duration is not uniform across the city; the darkest tracts cluster south and east of downtown.

[Figure: figures/joined_duration_choropleth.png]

Figure 10: Median review duration by tract income quintile. There is no strong monotonic pattern, which suggests duration differences track project type more than neighborhood wealth.

[Figure: figures/joined_income_vs_duration.png]

### Result Validity

For each test we state the null hypothesis, the test, and how we verified its assumptions on the real data.

RQ1. H0: residential and non-residential permits have the same median review duration. We ran a two-sided Mann-Whitney U test on 23,667 residential and 10,126 non-residential permits with valid non-negative durations. The test assumes independent samples and an ordered outcome, both satisfied here because every permit appears once per group and duration is a ranked quantity. The result rejects the null: U = 1.58e8, p < 0.001, with a residential median of 119 days against 60 for non-residential.

RQ3. H0: the number of review cycles is not monotonically associated with total duration. Spearman's rho = 0.512 with p < 0.001 on n = 33,793 permits rejects it. Spearman needs only a monotonic relationship rather than linearity, which matters because the cycle count is discrete.

RQ3. H0: the probability a review step ends in corrections is the same across review types. A raw contingency table over all 46 review types violated the chi-square assumption that expected cell counts reach five, so we collapsed the outcome to a binary corrections flag and grouped the 16 most common review types, leaving the rest as Other. On that table the minimum expected count is 1,748, far above the threshold, and the test rejects the null: chi2 = 19,027, dof = 15, p < 0.001.

## Challenge Goals Update

All five goals from the proposal are still in play, and the EDA work advanced four of them materially.

Multiple Datasets is now complete for the EDA stage: three datasets loaded, two joins performed, and every join combination summarized and plotted rather than only the individual tables.

Messy Data turned out to be even better suited than the proposal predicted. Beyond the API pulls we anticipated, the raw files delivered comma-formatted numerics, two incompatible timestamp styles, rows with unterminated quotes that break a strict parser, impossible negative durations, and census missing-value codes. Every one of these was handled programmatically.

Result Validity is complete for the EDA stage. All three hypothesis tests are run in code with their assumptions checked on the real data, and the chi-square test required a documented restructuring to satisfy its own assumptions.

Machine Learning stays on schedule for the final deliverable. The EDA established the feature pool (cost, housing units, cycles, complexity, class) and confirmed that cycle count correlates with duration, which is exactly the signal a predictive model will use.

New Library (SHAP) likewise remains a final-stage goal and is unchanged from the proposal.

## Work Plan Evaluation

The proposal budgeted 36 hours across seven tasks and allocated 15 of them to the three EDA tasks. Data acquisition and cleaning went faster than estimated because the Socrata exports download directly as CSV, but the parser failure from unterminated quotes cost roughly an hour of diagnosis, and deciding how to treat the 83 negative durations took deliberate discussion. We estimate the EDA tasks landed within an hour of the 15-hour budget. The remaining tasks, feature engineering and model training (8h), SHAP interpretation (4h), and the final report (5h), are unchanged.

## Testing

We test eda.py with test_eda.py, a set of assert-based tests that feed small hand-built inputs to each cleaning and summary helper and check exact outputs: comma strings convert to floats, both timestamp styles parse to the right year, the seven-number summary matches hand-computed values, missingness counts and percentages are per column, and value counts return a two-column table. We test the helpers this way because every figure and test in the report depends on them, so a bug there would silently corrupt everything downstream. The helpers run on inputs small enough to verify by hand, which is why the tests stay fast and independent of the full files. On top of the unit tests, the printed summaries allow a direct sanity check against the raw files, and the figures are cross-checked against each other, for example the tracts shaded darkest in the choropleth line up with the high end of the income-quintile boxplots.

## Collaboration

Mark Wang and William Wang divided the work as described in the Work Plan: Mark handled data acquisition, cleaning, and the statistical tests, while William handled the exploratory visualizations and the machine-learning preparation. Beyond the team, we discussed the project with a roommate who works as a data analyst at a large technology company, and he suggested the tract-level choropleth map and the income-quintile comparison as a way to make the geographic patterns visible.
