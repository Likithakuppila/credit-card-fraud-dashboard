
import os
import joblib
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="AI-Based Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 AI-Based Fraud Detection in Digital Payment Gateways")
st.write("Transaction Analysis and Machine Learning Predictions")

# Upload dataset
uploaded_file = st.file_uploader(
    "Upload your credit card dataset (CSV)",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Upload your creditcard.csv dataset to begin.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Unable to read CSV file: {e}")
    st.stop()

# Validate dataset
if not {"Class", "Amount"}.issubset(df.columns):
    st.error("Dataset must contain Class and Amount columns.")
    st.stop()

df["Class"] = pd.to_numeric(df["Class"], errors="coerce")
df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
df = df.dropna(subset=["Class", "Amount"]).reset_index(drop=True)

if df.empty:
    st.error("The uploaded dataset contains no valid transactions.")
    st.stop()

# Dashboard filters
st.sidebar.header("Dashboard Filters")

transaction_type = st.sidebar.selectbox(
    "Transaction Type",
    ["All", "Legitimate (0)", "Fraud (1)"]
)

max_amount = max(float(df["Amount"].max()), 1.0)

amount_range = st.sidebar.slider(
    "Amount Range",
    min_value=0.0,
    max_value=max_amount,
    value=(0.0, max_amount)
)

filtered = df[
    df["Amount"].between(amount_range[0], amount_range[1])
]

if transaction_type == "Legitimate (0)":
    filtered = filtered[filtered["Class"] == 0]
elif transaction_type == "Fraud (1)":
    filtered = filtered[filtered["Class"] == 1]

# Summary metrics
total = len(filtered)
fraud = int((filtered["Class"] == 1).sum())
legitimate = int((filtered["Class"] == 0).sum())

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Transactions", f"{total:,}")
c2.metric("Fraud Transactions", f"{fraud:,}")
c3.metric("Legitimate Transactions", f"{legitimate:,}")
c4.metric(
    "Fraud Percentage",
    f"{fraud / total * 100:.2f}%" if total else "0.00%"
)

# Charts
if total > 0:
    counts = (
        filtered["Class"]
        .map({0: "Legitimate", 1: "Fraud"})
        .value_counts()
        .rename_axis("Transaction Type")
        .reset_index(name="Count")
    )

    st.subheader("Fraud vs Legitimate Transactions")
    st.plotly_chart(
        px.bar(
            counts,
            x="Transaction Type",
            y="Count",
            color="Transaction Type"
        ),
        use_container_width=True
    )

    chart_data = filtered.sample(
        n=min(10000, total),
        random_state=42
    )

    st.subheader("Transaction Amount Distribution")
    st.plotly_chart(
        px.histogram(
            chart_data,
            x="Amount",
            color="Class",
            nbins=40
        ),
        use_container_width=True
    )

    st.subheader("Transaction Amount by Class")
    st.plotly_chart(
        px.box(
            chart_data,
            x="Class",
            y="Amount"
        ),
        use_container_width=True
    )

# AI Fraud Prediction
st.divider()
st.header("🤖 AI Fraud Prediction")

st.write(
    "Select a transaction from your dataset "
    "and test the trained machine learning models."
)

model_files = {
    "Random Forest": "random_forest_model.pkl",
    "XGBoost": "xgboost_model.pkl"
}

available_models = {
    name: path
    for name, path in model_files.items()
    if os.path.exists(path)
}

if not available_models:
    st.error(
        "Model files not found. Check that "
        "random_forest_model.pkl and xgboost_model.pkl "
        "are uploaded to the GitHub repository."
    )
    st.stop()

selected_model_name = st.selectbox(
    "Choose Machine Learning Model",
    list(available_models.keys())
)

row_number = st.number_input(
    "Transaction row number (starting from 0)",
    min_value=0,
    max_value=len(df) - 1,
    value=0,
    step=1
)

if st.button("Predict Transaction"):
    try:
        model = joblib.load(available_models[selected_model_name])

        # Match the feature columns used during training.
        transaction = df.drop(columns=["Class"]).iloc[
            [int(row_number)]
        ]

        # Predict transaction class.
        prediction = int(model.predict(transaction)[0])
        st.write("Debug — Predicted class:", prediction)
        st.write("Debug — Probability class order:", list(model.classes_))

        if prediction == 1:
            st.error("🚨 Model Prediction: FRAUD")
        else:
            st.success("✅ Model Prediction: LEGITIMATE")

        # Display probability and diagnostic information.
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(transaction)[0]
            classes = list(model.classes_)

            st.write("Debug — Model classes:", classes)
            st.write(
                "Debug — Raw probabilities:",
                [float(p) for p in probabilities]
            )

            if 1 in classes:
                fraud_index = classes.index(1)
                fraud_probability = float(
                    probabilities[fraud_index]
                )

                st.metric(
                    "Estimated Fraud Probability",
                    f"{fraud_probability * 100:.6f}%"
                )
            else:
                st.warning(
                    "Fraud class (1) was not found in model classes."
                )

        # Compare against the dataset's known label.
        actual = int(df.iloc[int(row_number)]["Class"])

        st.write(
            "Actual dataset label:",
            "Fraud (1)" if actual == 1 else "Legitimate (0)"
        )

        st.caption(
            "This demo predicts a selected transaction from the "
            "uploaded dataset. It does not connect to a real payment "
            "gateway or process live transactions."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info(
            "Check that the uploaded dataset has the same feature "
            "columns and order as the data used to train the model."
        )
