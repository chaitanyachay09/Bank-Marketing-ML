# Academic Task 2 — CSE570 Machine Learning with Python

## Title page
**Project:** Bank Marketing Subscription Prediction and Campaign Optimization Using Machine Learning  
**Student name:** [Your name]  
**Registration/Roll number:** [Your details]  
**Course:** CSE570 — Machine Learning with Python  
**Academic Task:** 2  
**Faculty:** [Faculty name]  
**Submission date:** [Confirm with faculty]

## 1. Problem statement (Q1 — 5 marks)
A Portuguese banking institution conducts direct marketing campaigns by phone to offer term deposits. The project develops a binary classification model to estimate whether a contacted customer subscribes (`y=yes`) using information available before the call. The intended use is campaign prioritization and analytical decision support, not automated exclusion of customers.

### Objectives
1. Profile and clean the public dataset.
2. Explore target distribution and feature relationships.
3. test relevant statistical associations.
4. Compare a baseline and four classifiers using leakage-safe validation.
5. Interpret model performance and limitations.

### Success criteria
A reproducible pipeline, documented CV comparison, independent test evaluation, minority-class metrics, model interpretation and transparent limitations. Insert actual results after running.

### Assumptions and constraints
- Historical campaign patterns may not generalize to a different bank or time period.
- `duration` is excluded because it is only known after the call.
- `unknown` is retained as a meaningful recorded category.
- Prediction is not causal inference and does not guarantee subscription.

## 2. Dataset description (Q1 — 5 marks)
Source: Moro, S., Rita, P., & Cortez, P. (2014), UCI Bank Marketing, https://doi.org/10.24432/C5K306.
- Rows: [from data_profile.json]
- Columns: [from data_profile.json]
- Input predictors: 16 before the deliberate exclusion of duration.
- Target: y (yes/no).
- Feature types: numeric/integer and categorical/binary.
- Target distribution: [copy counts and percentages from data_profile.json].
- Missing-value summary: [summarize missing_value_summary.csv].
- Unknown-category counts: [data_profile.json].
Explain why the dataset is suitable: public, real-world binary outcome, mixed data types, sufficient observations and documented provenance.

## 3. Methodology / workflow
Describe loading, profiling, duplicate handling, EDA, statistical testing, stratified split, ColumnTransformer, Pipeline, cross-validation, tuning, model selection and final test evaluation. Include a workflow diagram if desired.

## 4. Data cleaning and visualization (Q2)
Document duplicates, missing values, unknown categories, datatype checks, outlier inspection and why genuine extreme values were not automatically removed. Insert labelled figures from outputs.

## 5. EDA and statistical analysis (Q2)
### Univariate
Describe age, balance, campaign, pdays and target distributions.

### Bivariate
Compare subscription rates across job, education, housing, loan and other categories.

### Multivariate
Discuss numerical correlation and possible redundancy. Note that correlation is not causation.

### Statistical tests
Report chi-square(job vs y): chi-square=[ ], df=[ ], p=[ ], Cramér's V=[ ].
Report Mann–Whitney(balance by y): U=[ ], p=[ ], median yes=[ ], median no=[ ].
Interpret effect size and practical significance, not only p-values.

## 6. Model development and evaluation (Q2)
Describe baseline and four classifiers. Explain why `duration` was excluded. Explain preprocessing within folds to avoid leakage. Include the CV comparison table from `model_comparison_cv.csv`.

### Final held-out test result
Insert accuracy, precision, recall, F1, ROC-AUC, PR-AUC and confusion matrix from `final_test_metrics.json`. Explain false positives and false negatives in business terms.

### Feature interpretation
Use `permutation_importance.csv` and `06_permutation_importance.png`. State that feature importance is associational, not causal.

### Model selection
Justify the model selected by highest training-only 5-fold CV F1, then report its untouched test results. Discuss any trade-offs and whether a threshold change may be needed.

## 7. Limitations
- Historical and institution-specific dataset.
- Potential temporal and population drift.
- Class imbalance and threshold dependence.
- Some values are recorded as `unknown`.
- No causal conclusions.
- No guarantee of real-world campaign uplift.
- `duration` is excluded for pre-call use; results will differ from studies that include it.
- External validation on recent data is required before deployment.

## 8. Conclusion
Summarize the actual best model and metrics, main EDA findings, useful predictors and limitations. Do not insert fabricated results.

## 9. References
1. Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306
2. Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825–2830.
3. McKinney (2010). Data Structures for Statistical Computing in Python. Proceedings of the 9th Python in Science Conference.
4. Hunter (2007). Matplotlib: A 2D Graphics Environment. Computing in Science & Engineering, 9(3), 90–95.
5. Seaborn documentation: https://seaborn.pydata.org/
6. SciPy documentation: https://docs.scipy.org/doc/scipy/

## 10. GitHub and LinkedIn evidence
GitHub repository: [actual URL]
LinkedIn post: [actual URL]
Dated screenshot: [append]
Commit/PR evidence: [actual links]
