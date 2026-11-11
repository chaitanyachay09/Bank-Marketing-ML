
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "outputs" / "best_model.joblib"
METRICS_PATH = ROOT / "outputs" / "final_test_metrics.json"

# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🏦 Bank Marketing Subscription Prediction")
st.write(
    "Machine learning application to predict whether "
    "a customer is likely to subscribe to a term deposit."
)

st.divider()

if not MODEL_PATH.exists():
    st.error("Trained model not found. Run python src/train.py first.")
    st.stop()

try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()

# --------------------------------------------------
# CUSTOMER INPUT FORM
# --------------------------------------------------

st.subheader("Customer Information")
st.write("Enter the customer's details below.")

with st.form("customer_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age", min_value=18, max_value=100, value=35
        )

        job = st.selectbox(
            "Job",
            ["admin.", "blue-collar", "entrepreneur",
             "housemaid", "management", "retired",
             "self-employed", "services", "student",
             "technician", "unemployed", "unknown"]
        )

        marital = st.selectbox(
            "Marital Status",
            ["married", "single", "divorced", "unknown"]
        )

        education = st.selectbox(
            "Education",
            ["primary", "secondary", "tertiary", "unknown"]
        )

        default = st.selectbox(
            "Credit in Default",
            ["no", "yes", "unknown"]
        )

    with col2:
        balance = st.number_input(
            "Average Yearly Balance",
            min_value=-100000,
            max_value=1000000,
            value=1000
        )

        housing = st.selectbox(
            "Housing Loan",
            ["yes", "no", "unknown"]
        )

        loan = st.selectbox(
            "Personal Loan",
            ["yes", "no", "unknown"]
        )

        contact = st.selectbox(
            "Contact Communication Type",
            ["cellular", "telephone", "unknown"]
        )

        day = st.number_input(
            "Last Contact Day of Month",
            min_value=1, max_value=31, value=15
        )

    with col3:
        month = st.selectbox(
            "Last Contact Month",
            ["jan", "feb", "mar", "apr", "may", "jun",
             "jul", "aug", "sep", "oct", "nov", "dec"]
        )

        campaign = st.number_input(
            "Number of Contacts During Campaign",
            min_value=1, max_value=100, value=2
        )

        pdays = st.number_input(
            "Days Since Previous Contact (-1 = Never Contacted)",
            min_value=-1, max_value=999, value=-1
        )

        previous = st.number_input(
            "Number of Previous Contacts",
            min_value=0, max_value=100, value=0
        )

        poutcome = st.selectbox(
            "Previous Campaign Outcome",
            ["unknown", "failure", "other", "success"]
        )

    submitted = st.form_submit_button(
        "Predict Subscription",
        use_container_width=True
    )

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if submitted:

    customer = pd.DataFrame([{
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "day": day,
        "month": month,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome
    }])

    try:
        prediction = int(model.predict(customer)[0])

        st.divider()
        st.subheader("Prediction Result")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(customer)[0]
            classes = list(model.classes_)
            yes_index = classes.index(1)
            probability = float(probabilities[yes_index])

            st.metric(
                "Estimated Subscription Probability",
                f"{probability * 100:.2f}%"
            )

            st.progress(probability)

        if prediction == 1:
            st.success(
                "The model predicts that this customer "
                "is likely to subscribe to a term deposit."
            )
        else:
            st.warning(
                "The model predicts that this customer "
                "is unlikely to subscribe to a term deposit."
            )

        st.caption(
            "This is a machine-learning estimate, not a guarantee "
            "of customer behaviour."
        )

        with st.expander("View submitted customer information"):
            st.dataframe(customer, use_container_width=True)

    except Exception as error:
        st.error(f"Prediction failed: {error}")

# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

st.divider()
st.subheader("Model Performance")

if METRICS_PATH.exists():

    with open(METRICS_PATH, encoding="utf-8") as file:
        metrics = json.load(file)

    st.write("Selected Model:", metrics["selected_model"])

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Accuracy", f'{metrics["accuracy"]:.3f}')
    c2.metric("Precision", f'{metrics["precision"]:.3f}')
    c3.metric("Recall", f'{metrics["recall"]:.3f}')
    c4.metric("F1 Score", f'{metrics["f1"]:.3f}')

    st.write(
        "ROC-AUC:",
        f'{metrics["roc_auc"]:.3f}'
    )

    st.write("Model Evaluation Charts")

    chart_files = [
        ("Confusion Matrix", "03_confusion_matrix.png"),
        ("ROC Curve", "04_roc_curve.png"),
        ("Precision-Recall Curve", "05_precision_recall_curve.png"),
        ("Feature Importance", "06_permutation_importance.png")
    ]

    for title, filename in chart_files:
        image_path = ROOT / "outputs" / filename

        if image_path.exists():
            st.markdown(f"**{title}**")
            st.image(str(image_path), use_container_width=True)

else:
    st.info("Run the training script to generate model metrics.")

st.divider()
st.caption("CSE570 | Bank Marketing Machine Learning Project")
