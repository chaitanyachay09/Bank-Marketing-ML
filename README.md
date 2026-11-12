# Bank Marketing Subscription Prediction

A machine learning classification project that predicts whether a bank customer is likely to subscribe to a term deposit using historical marketing campaign data.

## 1. Project Overview

Banks conduct direct marketing campaigns to promote term deposit subscriptions. Identifying customers who are more likely to subscribe can help banks improve campaign efficiency and make better marketing decisions.

This project applies supervised machine learning techniques to analyze customer information and predict subscription outcomes.

## 2. Project Objectives

- Perform exploratory data analysis on bank marketing data.
- Identify missing and unknown values.
- Preprocess numerical and categorical features.
- Train and compare multiple classification algorithms.
- Evaluate model performance using classification metrics.
- Deploy an interactive prediction application using Streamlit.

## 3. Dataset Description

**Dataset:** UCI Bank Marketing Dataset (`bank-full`)

| Property | Details |
|---|---|
| Total records | 45,211 |
| Total columns | 17 |
| Input features | 16 |
| Target variable | `y` |
| Problem type | Binary classification |

**Target distribution:**

| Class | Records | Percentage |
|---|---:|---:|
| No | 39,922 | 88.3% |
| Yes | 5,289 | 11.7% |

The target variable represents whether a customer subscribed to a term deposit.

## 4. Data Preprocessing

The following preprocessing techniques were applied:

- Median imputation for numerical features.
- Standard scaling for numerical variables.
- Most-frequent imputation for categorical features.
- One-hot encoding for categorical variables.
- Handling of previously unseen categorical values.

The `duration` feature was excluded to prevent data leakage because it is only known after a customer interaction.

## 5. Machine Learning Models

The following algorithms were evaluated:

1. Dummy Classifier
2. Logistic Regression
3. Decision Tree
4. Random Forest
5. HistGradientBoosting

### Cross-Validation Results

Models were compared using five-fold cross-validation and F1-score.

| Model | Mean F1-score |
|---|---:|
| Dummy Classifier | 0.0000 |
| Logistic Regression | 0.3654 |
| Decision Tree | 0.3889 |
| Random Forest | 0.4608 |
| HistGradientBoosting | 0.3325 |

**Selected model: Random Forest**

Random Forest achieved the highest cross-validation F1-score among the evaluated models.

## 6. Final Model Evaluation

The selected Random Forest model achieved the following results on the held-out test dataset:

| Metric | Score |
|---|---:|
| Accuracy | 86.81% |
| Precision | 44.39% |
| Recall | 50.47% |
| F1-score | 47.24% |
| ROC-AUC | 79.11% |

These results indicate that the model can identify potential subscribers, although further improvements in precision and recall are possible.

## 7. Technologies Used

- Python – 3.15.2
- Pandas – 3.1.1
- NumPy – 2.4.2
- Matplotlib – 3.11.2
- Seaborn – 0.13.2
- Scikit-learn – 1.9.1
- Streamlit – 1.67.0
- Plotly – 7.1.0
- Jupyter Notebook – 7.6.3

## 8. Project Structure

```text
Bank-Marketing-ML/
│
├── docs/
├── outputs/
├── src/
├── app.py
├── analysis_walkthrough.ipynb
├── requirements.txt
└── README.md
```

## 9. Installation and Execution

### Clone the repository

```bash
git clone https://github.com/chaitanyachay09/Bank-Marketing-ML.git
cd Bank-Marketing-ML
```

### Create a virtual environment

```bash
python -m venv .venv
```

### Activate the environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit application

```bash
streamlit run app.py
```

## 10. Live Application

**Streamlit Deployment:**  
https://bank-marketingprediction.streamlit.app/

## 11. Future Enhancements

- Improve prediction performance for the minority class.
- Perform additional hyperparameter optimization.
- Explore more classification algorithms.
- Improve model interpretability.
- Evaluate performance on new customer data.

## 12. Conclusion

This project demonstrates how machine learning can support bank marketing decisions through customer subscription prediction. Multiple classification algorithms were evaluated, and Random Forest was selected based on its cross-validation F1-score.

The project highlights the importance of data preprocessing, class imbalance handling, data leakage prevention, and appropriate model evaluation.
