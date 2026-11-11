# CSE570 — Bank Marketing Subscription Prediction

**Task:** Binary classification — predict whether a bank customer subscribes to a term deposit.

**Dataset:** UCI Bank Marketing, dataset ID 222, `bank-full` release. Official source: https://archive.ics.uci.edu/dataset/222/bank  
**Citation:** Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing*. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306

## Important dataset clarification
The UCI page reports 45,211 instances and 17 total columns in the original `bank-full.csv`: 16 input predictors plus target `y`. Some UCI page versions display 16 features because they count predictors separately from the target. The downloaded data schema is profiled by the script.

The script excludes `duration` from the primary prediction task. UCI warns that call duration is only known after the call and should be discarded for realistic pre-call prediction.

## Fast start (VS Code terminal)
```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
```
Internet is required on first run to fetch the official dataset using `ucimlrepo`.

## Outputs
The script writes all generated artifacts to `outputs/`:
- data profile, missing-value summary, descriptive statistics and schema profile
- EDA figures and correlation heatmap
- chi-square and Mann–Whitney statistical test results
- five-fold CV model comparison
- final held-out test metrics and classification report
- confusion matrix, ROC curve, precision-recall curve
- permutation feature importance
- serialized selected model (`best_model.joblib`)

## Models
- DummyClassifier baseline
- Logistic Regression
- Decision Tree
- Random Forest
- HistGradientBoosting

The first four non-baseline classifiers satisfy the requirement for at least three predictive ML models. All preprocessing is inside sklearn Pipelines and is fitted only on training folds.

## Before submission
1. Run the script yourself and inspect every output.
2. Use the actual values from your output files in the report. Do not copy example or invented metrics.
3. Add your name, registration number, course details, faculty details and date to the report.
4. Create your own GitHub commits, pull request(s), LinkedIn post and dated evidence.
5. Be ready to explain the code and limitations in the individual viva.


## Full analysis demonstration
- Open `analysis_walkthrough.ipynb` in VS Code or Jupyter and run cells top-to-bottom.
  It explains dataset profiling, cleaning, univariate/bivariate/multivariate EDA,
  statistical tests, preprocessing, model lineup and result interpretation.
- Follow `docs/COMPLETE_ANALYSIS_AND_DEMONSTRATION_GUIDE.md` for what to show,
  what to explain to faculty, and a fill-in worksheet for actual findings.
