from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix, classification_report,
                             RocCurveDisplay, PrecisionRecallDisplay)
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     GridSearchCV, cross_validate)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
import joblib

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
RANDOM_STATE = 42
TARGET = "y"

def save_json(obj, path):
    path.write_text(json.dumps(obj, indent=2, default=lambda x: float(x)), encoding="utf-8")

def load_data():
    """Fetch official UCI dataset. Internet is required on first run."""
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=222)
    X = ds.data.features.copy()
    y = ds.data.targets.copy()
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    df = X.copy()
    df[TARGET] = y.astype(str).str.strip().str.lower().values
    # Standardize common column spelling in UCI releases.
    if "day_of_week" in df.columns:
        df = df.rename(columns={"day_of_week": "day"})
    return df

def cramers_v(table):
    chi2 = chi2_contingency(table)[0]
    n = table.to_numpy().sum()
    r, k = table.shape
    denom = n * min(k - 1, r - 1)
    return float(np.sqrt(chi2 / (n * (min(k, r) - 1)))) if denom > 0 else 0.0

def main():
    print("1/7 Downloading UCI Bank Marketing data...")
    df = load_data()
    print("Shape:", df.shape)
    print(df.head())

    # Data-quality profile. UCI marks some categorical values as 'unknown';
    # those are retained as an explicit category rather than silently deleted.
    duplicates = int(df.duplicated().sum())
    missing = df.isna().sum().sort_values(ascending=False)
    profile = {
        "rows": int(df.shape[0]),
        "columns_including_target": int(df.shape[1]),
        "predictor_columns_before_exclusions": int(df.shape[1] - 1),
        "duplicate_rows": duplicates,
        "missing_values_by_column": {k: int(v) for k, v in missing.items()},
        "target_counts": {k: int(v) for k, v in df[TARGET].value_counts().items()},
        "target_percentages": {k: float(v) for k, v in (df[TARGET].value_counts(normalize=True)*100).round(2).items()},
        "unknown_category_counts": {
            c: int((df[c].astype(str).str.lower() == "unknown").sum())
            for c in df.select_dtypes(include="object").columns if c != TARGET
        }
    }
    save_json(profile, OUT / "data_profile.json")
    missing.rename("missing_count").to_csv(OUT / "missing_value_summary.csv")
    df.describe(include="all").transpose().to_csv(OUT / "descriptive_statistics.csv")
    pd.DataFrame({"column": df.columns,
                  "dtype": [str(x) for x in df.dtypes],
                  "unique_values": [int(df[c].nunique(dropna=False)) for c in df.columns],
                  "missing_values": [int(df[c].isna().sum()) for c in df.columns]}).to_csv(
                      OUT / "data_dictionary_profile.csv", index=False)

    # Remove exact duplicate rows only; document count.
    df = df.drop_duplicates().copy()

    # EDA plots
    print("2/7 Creating EDA plots...")
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x=TARGET)
    plt.title("Target class distribution")
    plt.tight_layout(); plt.savefig(OUT/"01_target_distribution.png", dpi=160); plt.close()

    num_all = df.select_dtypes(include=np.number).columns.tolist()
    for col in ["age", "balance", "campaign", "pdays", "previous"]:
        if col in df.columns:
            plt.figure(figsize=(7,4))
            sns.histplot(data=df, x=col, hue=TARGET, bins=35, element="step", stat="density", common_norm=False)
            plt.title(f"Distribution of {col} by subscription")
            plt.tight_layout(); plt.savefig(OUT/f"eda_{col}_distribution.png", dpi=150); plt.close()

    for col in ["job", "education", "housing", "loan", "marital"]:
        if col in df.columns:
            rate = pd.crosstab(df[col], df[TARGET], normalize="index") * 100
            if "yes" in rate.columns:
                rate = rate.sort_values("yes", ascending=False)
            ax = rate.plot(kind="bar", stacked=True, figsize=(9,4))
            ax.set_ylabel("Within-category percentage")
            ax.set_title(f"Subscription distribution by {col}")
            plt.tight_layout(); plt.savefig(OUT/f"eda_{col}_subscription.png", dpi=150); plt.close()

    # Outlier inspection: boxplots are diagnostic only. Do not delete extreme
    # but potentially valid customer balances/contact counts automatically.
    for col in ["age", "balance", "campaign", "pdays", "previous"]:
        if col in df.columns:
            plt.figure(figsize=(7,3.5))
            sns.boxplot(data=df, x=col)
            plt.title(f"Outlier inspection: {col}")
            plt.tight_layout()
            plt.savefig(OUT/f"outlier_{col}_boxplot.png", dpi=150)
            plt.close()

    numeric = df.select_dtypes(include=np.number)
    if not numeric.empty:
        plt.figure(figsize=(10,7))
        sns.heatmap(numeric.corr(), cmap="coolwarm", center=0, annot=True, fmt=".2f")
        plt.title("Numerical feature correlation heatmap")
        plt.tight_layout(); plt.savefig(OUT/"02_numeric_correlation.png", dpi=160); plt.close()

    # Statistical analysis: association between job and subscription; balance
    # distribution difference between target groups. Interpret effect size too.
    print("3/7 Running statistical tests...")
    contingency = pd.crosstab(df["job"], df[TARGET]) if "job" in df.columns else pd.crosstab(df[TARGET], df[TARGET])
    chi2, p_chi, dof, _ = chi2_contingency(contingency)
    cv = cramers_v(contingency)
    if "balance" in df.columns:
        yes_bal = df.loc[df[TARGET] == "yes", "balance"].dropna()
        no_bal = df.loc[df[TARGET] == "no", "balance"].dropna()
        u_stat, p_mw = mannwhitneyu(yes_bal, no_bal, alternative="two-sided")
        med_yes, med_no = float(yes_bal.median()), float(no_bal.median())
    else:
        u_stat, p_mw, med_yes, med_no = np.nan, np.nan, np.nan, np.nan
    stats = {
        "chi_square_test": {"question": "Is job category associated with subscription?", "chi2": float(chi2), "degrees_of_freedom": int(dof), "p_value": float(p_chi), "cramers_v_effect_size": cv},
        "mann_whitney_u": {"question": "Does balance distribution differ between yes and no groups?", "u_statistic": float(u_stat), "p_value": float(p_mw), "median_balance_yes": med_yes, "median_balance_no": med_no},
        "interpretation_note": "A small p-value indicates evidence of association/difference, not causation. Report effect sizes and practical significance."
    }
    save_json(stats, OUT/"statistical_tests.json")

    # Modeling target. Explicitly exclude duration for realistic pre-call prediction.
    print("4/7 Preparing leakage-safe train/test split...")
    excluded = [TARGET, "duration"]
    X = df.drop(columns=[c for c in excluded if c in df.columns])
    y = (df[TARGET] == "yes").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features)
    ], remainder="drop")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"f1": "f1", "recall": "recall", "precision": "precision", "roc_auc": "roc_auc", "average_precision": "average_precision"}

    models = {
        "Dummy baseline": (DummyClassifier(strategy="most_frequent"), {}),
        "Logistic Regression": (LogisticRegression(max_iter=1500, class_weight="balanced", solver="liblinear"),
                               {"model__C": [0.1, 1.0, 5.0]}),
        "Decision Tree": (DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
                          {"model__max_depth": [5, 10, None], "model__min_samples_leaf": [5, 20]}),
        "Random Forest": (RandomForestClassifier(n_estimators=250, class_weight="balanced_subsample",
                                                  random_state=RANDOM_STATE, n_jobs=-1),
                          {"model__max_depth": [12, None], "model__min_samples_leaf": [2, 5],
                           "model__max_features": ["sqrt"]}),
        "HistGradientBoosting": (HistGradientBoostingClassifier(random_state=RANDOM_STATE),
                                 {"model__learning_rate": [0.05, 0.1], "model__max_iter": [100, 180],
                                  "model__max_leaf_nodes": [15, 31]})
    }

    # Baseline and model comparison via CV on training data only.
    print("5/7 Training and tuning models (this may take several minutes)...")
    rows = []
    fitted = {}
    for name, (estimator, grid) in models.items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])
        if grid:
            search = GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=-1,
                                  refit=True, verbose=0, error_score="raise")
            search.fit(X_train, y_train)
            best = search.best_estimator_
            best_params = search.best_params_
            best_cv_f1 = float(search.best_score_)
        else:
            best = pipe.fit(X_train, y_train)
            cv_scores = cross_validate(best, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
            best_params = {"strategy": "most_frequent"}
            best_cv_f1 = float(np.mean(cv_scores["test_f1"]))
        fitted[name] = best
        rows.append({"model": name, "cv_f1": best_cv_f1, "best_parameters": json.dumps(best_params)})
        print(f"  {name}: CV F1={best_cv_f1:.4f}")

    comparison = pd.DataFrame(rows).sort_values("cv_f1", ascending=False)
    comparison.to_csv(OUT/"model_comparison_cv.csv", index=False)

    # Select best non-baseline by CV F1. One final test evaluation only.
    candidates = comparison[comparison.model != "Dummy baseline"]
    selected_name = candidates.iloc[0]["model"]
    selected = fitted[selected_name]
    print("6/7 Final held-out test evaluation:", selected_name)
    y_pred = selected.predict(X_test)
    y_prob = selected.predict_proba(X_test)[:, 1] if hasattr(selected, "predict_proba") else selected.decision_function(X_test)
    metrics = {
        "selected_model": selected_name,
        "test_records": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "pr_auc_average_precision": float(average_precision_score(y_test, y_prob)),
        "confusion_matrix_labels_0_no_1_yes": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, target_names=["no", "yes"], output_dict=True, zero_division=0),
        "best_model_selection_rule": "Highest 5-fold cross-validated F1 among non-baseline models; test set used once for final reporting.",
        "excluded_feature": "duration",
        "excluded_feature_reason": "Known only after the call; would cause post-outcome leakage for pre-call prediction."
    }
    save_json(metrics, OUT/"final_test_metrics.json")
    joblib.dump(selected, OUT/"best_model.joblib")

    # Confusion matrix, ROC and PR curves.
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No","Yes"], yticklabels=["No","Yes"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"Confusion Matrix — {selected_name}")
    plt.tight_layout(); plt.savefig(OUT/"03_confusion_matrix.png", dpi=160); plt.close()

    RocCurveDisplay.from_predictions(y_test, y_prob)
    plt.title(f"ROC Curve — {selected_name}")
    plt.tight_layout(); plt.savefig(OUT/"04_roc_curve.png", dpi=160); plt.close()

    PrecisionRecallDisplay.from_predictions(y_test, y_prob)
    plt.title(f"Precision-Recall Curve — {selected_name}")
    plt.tight_layout(); plt.savefig(OUT/"05_precision_recall_curve.png", dpi=160); plt.close()

    # Permutation importance on held-out data; model remains untouched.
    print("7/7 Computing permutation importance and saving outputs...")
    perm = permutation_importance(selected, X_test, y_test, n_repeats=5,
                                  random_state=RANDOM_STATE, scoring="f1", n_jobs=-1)
    importance = pd.DataFrame({"feature": X_test.columns,
                               "importance_mean": perm.importances_mean,
                               "importance_std": perm.importances_std}).sort_values("importance_mean", ascending=False)
    importance.to_csv(OUT/"permutation_importance.csv", index=False)
    top = importance.head(15).sort_values("importance_mean")
    plt.figure(figsize=(8,6))
    plt.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
    plt.xlabel("Decrease in F1 after permutation")
    plt.title("Permutation Feature Importance")
    plt.tight_layout(); plt.savefig(OUT/"06_permutation_importance.png", dpi=160); plt.close()

    print("\nDONE. Open outputs/final_test_metrics.json and outputs/model_comparison_cv.csv.")
    print(json.dumps({k: metrics[k] for k in ["selected_model","accuracy","precision","recall","f1","roc_auc","pr_auc_average_precision"]}, indent=2))

if __name__ == "__main__":
    main()
