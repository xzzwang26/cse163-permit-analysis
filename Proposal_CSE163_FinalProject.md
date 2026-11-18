# Deconstructing Delays: Modeling Building Permit Review Duration in Seattle

Mark Wang and William Wang  
CSE 163, Autumn 2026  
Part 1: Project Proposal

## Research Questions

We plan to answer three questions about how building permits move through Seattle's review process.

1. How is a permit's total time from application to issuance distributed across the stages of plan review, and how does that distribution shift with permit type, project complexity, and neighborhood?
2. Which attributes of a permit, including its estimated cost, housing units, complexity class, and review history, most strongly predict how long review will take, and how accurately can we predict that duration for permits the model has not seen?
3. How much slower are permits that get sent back for corrections than permits approved on the first pass, and are those correction cycles concentrated in particular review types or teams?

## Motivation

Seattle's construction permitting process has a reputation for being slow, which feeds into the cost of housing and how quickly small businesses can open. In 2025, however, the city became one of the few large U.S. cities to publish machine-readable data covering its entire permitting pipeline, including individual review steps. This means that, for the first time, the delay can be measured and explained rather than simply complained about. Understanding which stages are slow and which factors contribute to delays is actionable. It informs applicants of what to expect and enables the city to identify areas for improvement. While a descriptive study would merely show that permits are slow, an explanatory model can show why. It can distinguish between 'this project is inherently complex' and 'this stage of the process is a bottleneck'. This distinction is what makes the analysis useful. The problem lies at the intersection of urban policy and applied machine learning, where every dataset is public and recent. This means the work can be reproduced and used outside the course.

## Dataset

We will use three public datasets, none of which appeared in a CSE 163 assignment or demo. Only the Census data needs a free API key; the other two download directly.

| Dataset              | What it contains                                                 | Key fields                                                                                                                | Access                                       |
| -------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| Building Permits     | All permits issued or in progress in Seattle, about 194,000 rows | PermitNum, PermitClassMapped, PermitTypeMapped, HousingUnitsAdded, EstProjectCost, AppliedDate, IssuedDate, StatusCurrent | <https://data.seattle.gov/dataset/76t5-zqzr> |
| Plan Review          | Individual review steps for each permit, about 127,000 rows      | PermitNum, ReviewCycle, ReviewType, ReviewTeam, ReviewerFinishDate, ReviewResultDesc, ReviewComplexityDesc                | <https://data.seattle.gov/d/tqk8-y2z5>       |
| ACS 5-Year Estimates | Census tract demographics for neighborhood controls              | Median household income (B19013), population (B01003), race (B02001)                                                      | <https://api.census.gov/data/2024/acs/acs5>  |

The two Seattle tables can be downloaded directly as CSV through the Socrata API. The Building Permits CSV lives at <https://cos-data.seattle.gov/api/v3/views/76t5-zqzr/export.csv?accessType=DOWNLOAD> and the Plan Review CSV at <https://cos-data.seattle.gov/api/v3/views/tqk8-y2z5/export.csv?accessType=DOWNLOAD>. The ACS tables can be pulled with the Census Data API after requesting a free key at <https://api.census.gov/data/key_signup.html>, or browsed at data.census.gov. All cleaning, joining, and analysis will be done programmatically in Python, never by hand.

## Challenge Goals

We are targeting five challenge goals.

**Multiple Datasets.** We combine three datasets and join them explicitly. The two Seattle tables join on **PermitNum**, and permit records join to census tracts for the ACS features. Two of our research questions draw on more than one dataset, which satisfies the minimum of three datasets, two multi-dataset questions, and one join operation.

**Messy Data.** The data does not arrive as a clean CSV. We pull it through the Socrata API and the Census Data API, both of which need preprocessing before they are usable. **ReviewResultDesc** and **ReviewComplexityDesc** are free text categories that have to be mapped, **EstProjectCost** has missing values that need handling, and the date fields are floating timestamps that must be converted before any duration can be computed.

**Result Validity.** We will back our claims with statistical tests instead of visual impression. A Mann-Whitney U test will check whether residential and non-residential permits differ in duration, since the durations are skewed rather than normal. A Spearman correlation will test the relationship between the number of correction cycles and total duration. For the model we will use k-fold cross-validation and report mean absolute error with a confidence interval, so the predictive result does not hinge on one lucky train-test split.

**Machine Learning.** We will go past the Decision Tree pipeline from class by training RandomForestRegressor and GradientBoostingRegressor to predict review duration, and we will tune hyperparameters with a grid search over the number of estimators, maximum depth, and learning rate. We will define the best model by held-out mean absolute error and inspect feature importances to find the most informative attributes.

**New Library.** We will learn SHAP, a library not covered in this course, to interpret the trained model. SHAP answers two of our research questions: which features drive duration the most, and whether correction-related features dominate the predictions.

## Method

For the first question we join Plan Review to Building Permits on **PermitNum**, a one-to-many join since a single permit has many review rows, and this join satisfies the Multiple Datasets goal. We convert the floating timestamps to datetimes and compute each review step's duration from **ReviewerAssignDate** to **ReviewerFinishDate**. Total duration runs from **AppliedDate** to **IssuedDate**, and we drop permits that are still in review because **IssuedDate** is null. We then group by **PermitClassMapped** and **PermitTypeMapped** and compare medians and interquartile ranges, and we join permits to ACS tracts to see how duration lines up with neighborhood income. If one stage or one permit type dominates the total, that stage is the bottleneck, and the neighborhood comparison tells us whether duration is socioeconomically patterned.

For the second question we build one row per permit. From Plan Review we aggregate the number of review rows, the number of distinct **ReviewCycle** values, the number of distinct **ReviewTeam** values, and a count of corrections taken from **ReviewResultDesc**. We merge these with Building Permits fields such as **EstProjectCost**, **HousingUnitsAdded**, **PermitClassMapped**, and **PermitTypeMapped**, and with ACS income and population. We impute missing costs with the median and one-hot encode the categorical columns. We then train RandomForestRegressor and GradientBoostingRegressor on 80 percent of permits, tune with grid search under 5-fold cross-validation, and evaluate on the held-out 20 percent, which is the model comparison and tuning required by the Machine Learning goal, with the cross-validation supporting Result Validity. A low held-out error means duration is predictable from permit attributes, and the top features tell us which attributes drive slowness, which connects back to the bottleneck found in the first question.

For the third question we use **ReviewResultDesc** to split permits into those approved on the first pass and those sent back for corrections at least once. We compare their total durations with a Mann-Whitney U test and report the p-value and an effect size, and we group correction rows by **ReviewType** and **ReviewTeam** and use a chi-square test to see whether corrections concentrate in specific types. These tests are what satisfy the Result Validity goal. SHAP values then show whether correction-related features are among the strongest predictors, which ties into the New Library goal and either confirms or complicates the statistical result.

## Work Plan

We divided the work into seven tasks. Mark owns data cleaning and the statistical tests, and William owns exploratory analysis, the machine-learning modeling, and the SHAP interpretation. They split feature engineering and the report, so each person contributed about half of the roughly 36 hours.

| # | Task                                                                             | Hours | Owner                             |
| - | -------------------------------------------------------------------------------- | ----- | --------------------------------- |
| 1 | Acquire the three datasets and verify row counts and columns                     | 4     | Both                              |
| 2 | Clean and merge: parse timestamps, normalize text, join Seattle tables, join ACS | 6     | Mark                              |
| 3 | Exploratory analysis and visualizations for the first question                   | 5     | William                           |
| 4 | Statistical validity checks: Mann-Whitney U, chi-square, Spearman                | 4     | Mark                              |
| 5 | Feature engineering and model training with grid search and cross-validation     | 8     | William (modeling), Mark (features) |
| 6 | Model interpretation with SHAP                                                   | 4     | William                           |
| 7 | Write the report, tidy the code, verify reproducibility                          | 5     | Both                              |

The code lives in a private GitHub repository so changes are versioned and reviewable. We set up a local Python environment in VSCode with pandas, matplotlib, seaborn, scikit-learn, and shap, and we develop runnable scripts rather than relying on a notebook. Each analysis step is written as a function with a small hand-checked example so we can verify each other's intermediate results. If a task turns out harder than expected, the fallback keeps the challenge goals intact: we can use Seattle's built-in Neighborhood field instead of the ACS join while still holding three datasets, and we can limit SHAP to global feature importance rather than per-permit explanations.
