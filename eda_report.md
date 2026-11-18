# Exploratory Data Analysis: Modeling Building Permit Review Duration in Seattle

Mark Wang and William Wang
CSE 163, Autumn 2026
Part 2: Exploratory Data Analysis
GitHub repository: https://github.com/xzzwang26/cse163-permit-analysis

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
| Building Permits (data.seattle.gov/dataset/76t5-zqzr) | 194,311 x 41 | One building permit application | PermitNum, PermitClassMapped, PermitTypeMapped, EstProjectCost, TotalDaysPlanReview, DaysInitialPlanReview, DaysPlanReviewCity, DaysOutCorrections, NumberReviewCycles, HousingUnitsAdded, AppliedDate, IssuedDate, Latitude, Longitude, OriginalZip, StatusCurrent |
| Plan Review (data.seattle.gov/d/tqk8-y2z5) | 271,104 x 46 | One review step within one permit | PermitNum, ReviewCycle, ReviewType, ReviewTeam, ReviewResultDesc, ReviewComplexityDesc, ReviewerAssignDate, ReviewerFinishDate |
| ACS 5-Year Estimates (api.census.gov, 2024 acs5) | 495 x 9 | One census tract in King County | GEOID, median household income (B19013), population (B01003), five race count columns |

During the EDA we found that the raw files are messier than the proposal assumed. The numeric duration fields arrive as comma-formatted strings ('4,318'), the two Seattle tables use different date formats ('2006-09-29' versus '2018 Aug 31 12:00:00 AM'), the Plan Review CSV contains a handful of rows with unterminated quotes that break a strict parser, and the permits table holds 83 negative durations (60 of them on permits whose status is Completed) plus a few negative values in DaysInitialPlanReview, DaysPlanReviewCity and in the reviewer date fields, all clearly entry errors. We report these as raw-file findings. Every summary, figure, and test after the Dataset section uses the cleaned data with negative durations removed, and the text marks which population each number describes.

## Method

The EDA proceeded in six steps. First, we loaded only the columns each research question needs, which keeps the two large Seattle files manageable in memory. Second, we cleaned each table: comma-formatted numerics were stripped and converted, both timestamp styles were parsed, the census missing-value codes were set to NaN, and the 83 negative durations were set aside. Third, we produced the raw-file summaries the report requires, then verified the missingness explanation by comparing duration-missing and duration-present permits across status, application year, and permit type. Fourth, we built the two joins our Multiple Datasets goal requires, permits to reviews on PermitNum and permits to census tracts through a spatial join on permit coordinates, and summarized each joined table. Fifth, we decomposed the total duration into its stages and compared individual review steps across types. Sixth, we ran the hypothesis tests attached to our Result Validity goal, checking each test's assumptions on the actual data.

## EDA Results

### Building Permits

The permits table has 194,311 rows and 41 raw columns, of which we use 16. Each row is one application, and the columns describe the project (class, type, cost, housing units), its timeline (applied, issued), its review history (total days, days in the initial review, days waiting outside corrections, city-held days, cycle count), and its location (latitude, longitude, ZIP).

Missing data is structural rather than random, and checking the missingness against the data changed the explanation we proposed. **TotalDaysPlanReview** is absent for 82.6% of rows. By application year, permits filed from 2005 through 2013 carry the field 0% of the time, 2017 sits at 8%, and the rate jumps to 42% in 2018 and settles around 60% from 2020 onward, which points to the city only starting to populate these duration fields in 2018. By permit type, categories that never enter plan review, such as ECA and shoreline exemptions (18,937 rows) and roof work (2,133 rows), are missing 100% of the time, while Building (155,941 rows) and Demolition sit near 80% missing. By status, applications still moving through intake show no duration as expected, but even Completed permits carry one only 16.5% of the time, and Issued permits reach 69%. Taken together, the 33,794 permits with a usable duration are mostly Building or Demolition applications filed since 2018 whose process ran far enough to record timings. Every conclusion below describes that population rather than all permit activity in Seattle.

The raw duration column also holds 83 negative values (minimum -743), 60 of them on Completed permits, which cannot represent a real elapsed time. On the raw data the seven-number summary of **TotalDaysPlanReview** is mean 170.12, std 233.70, min -743, Q1 44, median 96, Q3 206, max 4,734. After removing the negatives the summary barely moves: mean 170.69, std 233.60, min 0, Q1 44, median 97, Q3 207, max 4,734, so the errors shift the location very little even though they matter for interpretation.

**PermitClassMapped** splits into 23,667 Residential and 10,126 Non-Residential permits within the cleaned sample, with one blank.

Figure 1: Distribution of permit review durations on a log-scaled count axis, for the 33,794 permits with non-negative durations. The distribution is heavily right-skewed, with a mode under 100 days and a long tail past 1,000.

[Figure: figures/permits_duration_hist.png]

Figure 2: Review duration by permit class. Residential permits run visibly longer than non-residential ones, a pattern the Mann-Whitney test below confirms.

[Figure: figures/permits_class_boxplot.png]

Figure 3: The most common permit types by application count.

[Figure: figures/permits_type_counts.png]

The stage columns decompose the total. The median first round of plan review (**DaysInitialPlanReview**) takes 27 days, the median wait outside corrections (**DaysOutCorrections**) is 23 days, and the median time the city holds the application (**DaysPlanReviewCity**) is 55 days against a median total of 97. The remaining days sit with the applicant between cycles. At the level of individual review steps (measured in the Plan Review table as the days from reviewer assignment to reviewer finish), steps have a median of 8 days, and the slowest large review types are Floodplain (median 19 days), Geo Soils (17), Housing (16) and ECA GeoTech (15), with Drainage and Structural Engineer near 12 to 13.

### Plan Review

The reviews table has 271,104 rows and 46 raw columns, of which we use 13 plus one derived column, the days a single step spent with its reviewer. Each row is one review step performed by one team on one permit, so a permit that cycles through corrections several times contributes many rows. The duration totals stored on these rows are permit-level fields repeated on every step, so the step-level quantitative variables are **ReviewCycle** and **step_days**.

**ReviewCycle** has a median of 2 across steps (max 15). **step_days** has a median of 8 days (mean 14, max 1,669) but also holds negative values down to -467, the same date-entry-error pattern seen in the permits table; the by-type comparison below excludes them.

**ReviewResultDesc** has four values, and their sizes already matter for the third research question: Corrections Required appears 139,571 times against 128,920 Approvals, meaning more than half of all review steps send the applicant back.

Figure 4: Review outcomes across all plan review steps.

[Figure: figures/reviews_result_counts.png]

Figure 5: The most common review types by step count. Addressing, Structural, and similar specialist reviews dominate.

[Figure: figures/reviews_type_counts.png]

### ACS Tracts

The census table has 495 rows, one per King County tract, with median household income, population, and five race counts. The median tract income is 128,447 dollars, and only two tracts have a missing income value (0.4%), which we set to NaN after replacing the census missing codes.

Figure 6: Distribution of tract median household income. Tracts range from under 30,000 dollars to over 250,000.

[Figure: figures/acs_income_hist.png]

Figure 7: Racial composition of tracts by income quintile. Aggregating the 495 tracts into five income groups makes the pattern readable, and the non-white share of population falls steadily from the lowest to the highest income quintile, which motivates treating tract demographics as a neighborhood control.

[Figure: figures/acs_race_composition.png]

### Join 1: Permits to Reviews

Because one permit goes through many review steps, the inner join on **PermitNum** keeps 249,401 review rows from the 33,794 cleaned permits and attaches the permit-level class and duration to each step. Every review row found a matching permit, so the join loses nothing. The median permit carries 9 review steps and the 90th percentile carries 24.

Figure 8: Share of review steps requiring corrections, by review cycle. Sixty percent of first-cycle steps come back with corrections, and the rate falls steadily to about 36% by the fifth cycle.

[Figure: figures/joined_cycle_correction_rate.png]

Reading the cycle dimension directly changed the story we had expected. The first cycle carries most of the correction burden: 59.6% of first-cycle steps require corrections, falling to 46.5% at cycle two, 42.2% at cycle three and 35.5% by cycle five. A permit that survives into a later cycle is reviewing a narrower set of remaining problems, so its chance of clearing that cycle rises. The first pass is where most of the back-and-forth sits.

### Join 2: Permits to Census Tracts

We joined permits to the 2020 tract geometry by point-in-polygon on the permit coordinates. 33,737 permits (100% of those with coordinates) landed inside a King County tract, and the ACS table then attaches income and demographics to every one of them.

Figure 9: Median review duration by census tract, over the cleaned sample. Duration is not uniform across the city; the darkest tracts cluster south and east of downtown.

[Figure: figures/joined_duration_choropleth.png]

Figure 10: Median review duration by tract income quintile. There is no strong monotonic pattern, which suggests duration differences track project type more than neighborhood wealth.

[Figure: figures/joined_income_vs_duration.png]

### Result Validity

For each test we state the null hypothesis, the test, the assumption checks on the real data, and any limitations that remain.

RQ1. H0: the duration distributions of residential and non-residential permits are identical. We ran a two-sided Mann-Whitney U test on 23,667 residential and 10,126 non-residential permits with valid non-negative durations. The test assumes independent samples and an ordered outcome, both satisfied here, and it evaluates whether one distribution tends to produce larger values than the other; it is not a test of medians unless the two shapes are also assumed equal, so we treat the medians as description. The result rejects the null (U = 1.58e8, p < 0.001). Descriptively the residential median is 119 days against 60 for non-residential.

RQ3. H0: the number of review cycles is not monotonically associated with total duration. Spearman's rho = 0.512 with p < 0.001 on n = 33,793 permits rejects it. Spearman needs only a monotonic relationship rather than linearity, which matters because the cycle count is discrete. One caution carries forward to the modeling stage: the cycle count is a process variable that only exists after review has started, so this association describes the review record and cannot serve as an at-application-time predictor.

RQ3. H0: permits that go through at least two review cycles have the same duration distribution as permits that clear in one. On 26,970 revised and 6,823 first-pass permits, the two-sided Mann-Whitney U test rejects the null (U = 1.18e8, p < 0.001). The revised median of 102.5 days is about 2.6 times the first-pass median of 39 days, which answers the 'how much slower' question directly: a permit that gets sent back spends roughly two more months in review than one approved on the first pass.

RQ3. H0: the probability that a review's first pass ends in corrections is the same across review types. Because one permit generates many dependent review rows, the unit of analysis here is each permit-review-type pair's first result, which keeps repeat cycles of the same permit from being counted as separate evidence. Review types outside the top 15 by volume fold into Other, giving 16 cells with a minimum expected count of 491.8, well above the chi-square threshold. One limitation remains: results on different review types within the same permit are still related, and the test treats them as independent, so the chi-square statistic is best read as evidence of strong aggregation rather than an exact p-value. The null is rejected (chi2 = 21,504, dof = 15, p < 0.001). Drainage reviews come back with corrections 91.5% of the time and Land Use 89.3%, against 30.9% for Addressing and 25.7% for City Light.

## Challenge Goals Update

All five goals from the proposal remain, and the EDA work advanced four of them materially.

**Multiple Datasets** is complete for the EDA stage: three datasets loaded, two joins performed, and each join combination summarized and plotted rather than only the individual tables.

**Messy Data** fit the project more closely than the proposal predicted. Beyond the API pulls we anticipated, the raw files delivered comma-formatted numerics, two incompatible timestamp styles, rows with unterminated quotes that break a strict parser, impossible negative durations in several fields, and census missing-value codes. Every one of these was handled programmatically.

**Result Validity** is complete for the EDA stage, with two upgrades over the proposal. The chi-square test was rebuilt on permit-review-type pairs after the row-level version failed its independence assumption, and a fourth test was added to answer the 'how much slower' question directly rather than only through the cycle count.

**Machine Learning** moves to the final deliverable with a scoping decision the EDA forced. The prediction time point has to be fixed first: a model that forecasts duration at application time may only use features known then, such as permit class, type, estimated cost, housing units and tract demographics. **NumberReviewCycles** and **DaysOutCorrections** correlate strongly with duration, but they only exist after review has started, so they would leak the future into an application-time model; they are appropriate for a second, mid-process model that updates the forecast once review is underway. The EDA established both feature sets and showed the outcome variable is well defined for 33,794 permits.

**New Library** (SHAP) likewise remains a final-stage goal and is unchanged from the proposal.

## Work Plan Evaluation

The proposal budgeted 36 hours across seven tasks and allocated 15 of them to the three EDA tasks. Data acquisition and cleaning went faster than estimated because the Socrata exports download directly as CSV, but the parser failure from unterminated quotes cost roughly an hour of diagnosis, and a round of review feedback on a report draft added several more hours of verification work: checking the missingness explanation against the data, rebuilding the chi-square test on an independent unit, and answering the corrections question with a direct duration comparison. We estimate the EDA tasks landed several hours over the 15-hour budget. The remaining tasks, feature engineering and model training (8h), SHAP interpretation (4h), and the final report (5h), are unchanged, and the model will now be built on the cleaned 33,794-permit sample with the application-time feature set described above.

## Testing

We test eda.py with test_eda.py, a set of assert-based tests that feed small hand-built inputs to each cleaning and summary helper and check exact outputs: comma strings convert to floats, both timestamp styles parse to the right year, the seven-number summary matches hand-computed values, missingness counts and percentages are per column, and value counts return a two-column table. We test the helpers this way because every figure and test in the report depends on them, so a bug there would silently corrupt everything downstream. The helpers run on inputs small enough to verify by hand, which is why the tests stay fast and independent of the full files. On top of the unit tests, the printed summaries allow a direct sanity check against the raw files, and the figures are cross-checked against each other and against the printed test statistics, for example the tracts shaded darkest in the choropleth line up with the high end of the income-quintile boxplots, and the correction-rate bar chart matches the chi-square table's rates.

## Collaboration

Mark Wang and William Wang divided the work as described in the Work Plan: Mark handled data acquisition, cleaning, and the statistical tests, while William handled the exploratory visualizations and the machine-learning preparation. Beyond the team, we discussed the project with a roommate who works as a data analyst at a large technology company, and he suggested the tract-level choropleth map and the income-quintile comparison as a way to make the geographic patterns visible. His review of a draft also pushed us to verify the missingness explanation and to rebuild the chi-square test on an independent unit.
