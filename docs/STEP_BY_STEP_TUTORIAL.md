# Step-by-step execution tutorial

## Step 1 — Install software
Install Python 3.10+ and VS Code. In VS Code install the Python extension. Extract the ZIP and open the `CSE570_Bank_Marketing_Project` folder.

## Step 2 — Create and activate a virtual environment
Open Terminal > New Terminal in VS Code.

Windows PowerShell:
```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
```
If script activation is blocked, use Command Prompt:
```bat
.venv\\Scripts\\activate.bat
```
macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Step 3 — Install libraries
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Step 4 — Run the complete project
From the project root:
```bash
python src/train.py
```
The first run downloads UCI dataset ID 222, so internet is required. Model tuning can take several minutes. Do not close the terminal until `DONE` appears.

## Step 5 — Check generated files
Open the `outputs` folder. Read these first:
1. `data_profile.json` — row count, columns, target distribution, missingness and unknown categories.
2. `descriptive_statistics.csv` — descriptive statistics.
3. `statistical_tests.json` — chi-square and Mann–Whitney tests.
4. `model_comparison_cv.csv` — cross-validation model comparison.
5. `final_test_metrics.json` — final held-out test results.
6. PNG files — EDA and model evaluation plots.
7. `permutation_importance.csv` — feature interpretation.

## Step 6 — Understand the problem
The bank wants to identify customers more likely to subscribe to a term deposit. The target `y` is binary: `yes` or `no`. This is classification, not regression.

### Objectives
- Build a reproducible pre-call subscription prediction pipeline.
- Compare a majority-class baseline with at least three classifiers.
- Evaluate minority-class performance, not accuracy alone.
- Identify useful predictors without claiming causation.
- Translate model outputs into campaign prioritization insights.

### Success criteria
- A complete, leakage-safe workflow.
- Cross-validated comparison and held-out test metrics.
- Report precision, recall, F1, ROC-AUC, PR-AUC and confusion matrix.
- Explain false negatives, false positives, limitations and business trade-offs.
Do not promise a particular score before running the experiment.

## Step 7 — Dataset and data dictionary
The UCI `bank-full` dataset contains 45,211 rows. It has 16 input predictors plus target `y` (17 columns in total). The script saves the observed schema profile.

| Variable | Meaning | Type |
|---|---|---|
| age | Customer age | Numeric |
| job | Occupation category | Categorical |
| marital | Marital status | Categorical |
| education | Education level | Categorical |
| default | Credit in default? | Binary categorical |
| balance | Average yearly balance (euros) | Numeric |
| housing | Housing loan? | Binary categorical |
| loan | Personal loan? | Binary categorical |
| contact | Contact communication type | Categorical |
| day | Day of month of last contact | Numeric/date-like |
| month | Month of last contact | Categorical |
| duration | Last call duration in seconds | Numeric; excluded from pre-call model |
| campaign | Number of contacts in current campaign | Numeric |
| pdays | Days since previous campaign contact; -1 means not previously contacted | Numeric/sentinel |
| previous | Number of contacts before current campaign | Numeric |
| poutcome | Previous campaign outcome | Categorical |
| y | Subscribed term deposit? | Binary target |

`unknown` is a literal category in this dataset and is not automatically the same as a missing NaN. The profile reports both NaN and `unknown` counts. The preprocessing pipeline imputes actual missing values but retains the `unknown` category.

## Step 8 — Cleaning and preprocessing
- Exact duplicate rows are counted and removed.
- Missing values are summarized.
- Numeric missing values are median-imputed inside the pipeline.
- Categorical missing values are most-frequent-imputed inside the pipeline.
- Categorical features are one-hot encoded.
- Numeric features are standardized.
- Outliers are inspected with distributions and boxplots/histograms; extreme values are not automatically deleted because they may be genuine customers. Discuss the decision.
- `duration` is excluded to prevent post-call leakage.
- `pdays = -1` is retained as a documented sentinel meaning no previous contact.

## Step 9 — EDA and statistical analysis
The script generates:
- target class distribution;
- numerical distributions by target;
- category-wise subscription composition;
- numerical correlation heatmap.

Statistical tests:
1. Chi-square test of independence: job category vs subscription. Report chi-square, degrees of freedom, p-value and Cramér's V.
2. Mann–Whitney U: balance distribution for subscribers vs non-subscribers. Report U, p-value and group medians.

Interpretation: p < 0.05 is evidence against the relevant null hypothesis under test assumptions; it does not show causation. With a large dataset, tiny differences can be statistically significant, so discuss effect size and practical significance.

## Step 10 — Models
- DummyClassifier: majority-class baseline.
- Logistic Regression: linear probabilistic baseline classifier.
- Decision Tree: interpretable nonlinear splits.
- Random Forest: bagged tree ensemble.
- HistGradientBoosting: sequential boosting of decision trees.

The script uses 5-fold StratifiedKFold on training data and compact GridSearchCV grids. Model selection is based on mean cross-validated F1 among non-baseline models. The test set is not used for tuning.

## Step 11 — Evaluation
The positive class is `yes`. Interpret:
- Precision: among predicted subscribers, fraction actually subscribing.
- Recall: among actual subscribers, fraction identified.
- F1: harmonic mean of precision and recall.
- ROC-AUC: ranking ability across thresholds.
- PR-AUC / average precision: useful under class imbalance.
- Confusion matrix: TN, FP, FN, TP.

A false negative is a potential subscriber the model misses; a false positive is a customer prioritized who does not subscribe. The business cost of each error should guide threshold choice. This starter implementation uses the default 0.5 threshold; threshold optimization can be a documented extension using validation data only.

## Step 12 — Report and evidence
Use `REPORT_TEMPLATE.md`. Insert your actual metrics and figures from `outputs/`. Cite UCI and libraries. Do not invent results, repository stars, likes or comments.

## Troubleshooting
- `ModuleNotFoundError`: activate `.venv`, then `pip install -r requirements.txt`.
- Download failure: check internet and rerun.
- PowerShell activation blocked: use Command Prompt activation.
- Script run from wrong folder: `cd` into the extracted project root first.
- Slow training: allow time; close other heavy applications. Do not remove the test split or CV to make scores look better.
