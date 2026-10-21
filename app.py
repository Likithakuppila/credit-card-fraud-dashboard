import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os

st.set_page_config(
    page_title="AI-Based Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 AI-Based Fraud Detection in Digital Payment Gateways")
st.subheader("Transaction Analysis and Machine Learning Predictions")

st.divider()

# ==================================================
# 1. UPLOAD DATASET
# ==================================================

st.header("📂 Upload Transaction Dataset")

uploaded_file = st.file_uploader(
    "Upload creditcard.csv",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload your creditcard.csv dataset.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as error:
    st.error(f"CSV loading error: {error}")
    st.stop()

if df.empty:
    st.error("The uploaded dataset is empty.")
    st.stop()

if "Class" not in df.columns:
    st.error("The dataset must contain a Class column.")
    st.stop()

# Convert target values to numeric
df["Class"] = pd.to_numeric(df["Class"], errors="coerce")

valid_df = df[df["Class"].isin([0, 1])].copy()

if valid_df.empty:
    st.error("No valid Class values (0 or 1) were found.")
    st.stop()

# ==================================================
# 2. DATASET SUMMARY
# ==================================================

st.header("📊 Transaction Overview")

total_transactions = len(df)
legitimate_count = int((valid_df["Class"] == 0).sum())
fraud_count = int((valid_df["Class"] == 1).sum())

classified_count = legitimate_count + fraud_count

fraud_percentage = (
    fraud_count / classified_count * 100
    if classified_count > 0 else 0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Transactions", f"{total_transactions:,}")
col2.metric("Legitimate Transactions", f"{legitimate_count:,}")
col3.metric("Fraud Transactions", f"{fraud_count:,}")
col4.metric("Fraud Percentage", f"{fraud_percentage:.4f}%")

with st.expander("Dataset Diagnostic Information"):
    st.write("Dataset shape:", df.shape)
    st.write(
        "Class values:",
        df["Class"].value_counts(dropna=False).to_dict()
    )
    st.write("Dataset columns:", df.columns.tolist())

with st.expander("View Dataset"):
    st.dataframe(df.head(10), use_container_width=True)

# ==================================================
# 3. TRANSACTION CHARTS
# ==================================================

st.divider()
st.header("📈 Transaction Analysis")

chart_df = pd.DataFrame({
    "Transaction Type": ["Legitimate", "Fraud"],
    "Count": [legitimate_count, fraud_count]
})

fig = px.bar(
    chart_df,
    x="Transaction Type",
    y="Count",
    color="Transaction Type",
    text="Count",
    title="Fraud vs Legitimate Transactions"
)

st.plotly_chart(fig, use_container_width=True)

if "Amount" in df.columns:

    fig_amount = px.histogram(
        valid_df,
        x="Amount",
        nbins=50,
        title="Transaction Amount Distribution"
    )

    st.plotly_chart(fig_amount, use_container_width=True)

    amount_chart_df = valid_df.copy()

    amount_chart_df["Transaction Type"] = (
        amount_chart_df["Class"].map({
            0: "Legitimate",
            1: "Fraud"
        })
    )

    fig_box = px.box(
        amount_chart_df,
        x="Transaction Type",
        y="Amount",
        color="Transaction Type",
        title="Transaction Amount by Class"
    )

    st.plotly_chart(fig_box, use_container_width=True)

# ==================================================
# 4. LOAD MACHINE LEARNING MODELS
# ==================================================

st.divider()
st.header("🤖 AI Fraud Prediction")

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
        "Model files not found. Please upload "
        "random_forest_model.pkl and xgboost_model.pkl "
        "to your GitHub repository."
    )
    st.stop()

selected_model_name = st.selectbox(
    "Select Machine Learning Model",
    list(available_models.keys())
)

row_number = st.number_input(
    "Transaction Row Number (starts from 0)",
    min_value=0,
    max_value=len(df) - 1,
    value=0,
    step=1
)

# ==================================================
# 5. PREDICTION
# ==================================================

if st.button("🔍 Analyze Transaction", type="primary"):

    try:
        model_path = available_models[selected_model_name]
        model = joblib.load(model_path)

        # Prepare features
        X = df.drop(columns=["Class"]).copy()

        transaction = X.iloc[[int(row_number)]].copy()

        # Match feature order from model training
        if hasattr(model, "feature_names_in_"):

            expected_features = list(model.feature_names_in_)

            missing_features = [
                feature
                for feature in expected_features
                if feature not in transaction.columns
            ]

            if missing_features:
                st.error(
                    "Missing model features: "
                    + ", ".join(missing_features)
                )
                st.stop()

            transaction = transaction[expected_features]

        # Ensure numeric input
        transaction = transaction.apply(
            pd.to_numeric,
            errors="coerce"
        )

        if transaction.isnull().any().any():
            st.error(
                "Selected transaction contains missing or "
                "non-numeric values."
            )
            st.stop()

        # Get prediction and probabilities
        prediction = int(model.predict(transaction)[0])

        probabilities = model.predict_proba(transaction)[0]

        model_classes = list(model.classes_)

        # Locate the fraud class
        fraud_indices = [
            index
            for index, label in enumerate(model_classes)
            if int(label) == 1
        ]

        if not fraud_indices:
            st.error("Fraud class 1 is missing from this model.")
            st.stop()

        fraud_index = fraud_indices[0]

        # Extract fraud probability
        fraud_probability = float(
            probabilities[fraud_index]
        )

        # ==================================================
        # 6. DISPLAY RESULT
        # ==================================================

        st.subheader("🔍 Prediction Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            if prediction == 1:
                st.error("🚨 FRAUD DETECTED")
            else:
                st.success("✅ LEGITIMATE TRANSACTION")

        with result_col2:

            st.metric(
                "Estimated Fraud Probability",
                f"{fraud_probability * 100:.8f}%"
            )

        st.write("Fraud Probability Indicator")

        st.progress(
            min(max(fraud_probability, 0.0), 1.0)
        )

        # ==================================================
        # 7. DEBUG INFORMATION
        # ==================================================

        st.subheader("🛠️ Model Diagnostic Information")

        st.write("Selected Model:", selected_model_name)
        st.write("Model File:", model_path)
        st.write("Transaction Row:", int(row_number))
        st.write("Prediction:", prediction)
        st.write("Model Classes:", model_classes)
        st.write("Raw Probabilities:", probabilities.tolist())
        st.write("Fraud Class Index:", fraud_index)
        st.write("Fraud Probability (decimal):", fraud_probability)
        st.write(
            "Fraud Probability (percentage):",
            fraud_probability * 100
        )

        highest_probability_class = int(
            model_classes[int(probabilities.argmax())]
        )

        st.write(
            "Class with Highest Probability:",
            highest_probability_class
        )

        if prediction != highest_probability_class:
            st.warning(
                "Prediction and highest-probability class differ. "
                "Check the model and its probability output."
            )
        else:
            st.success(
                "Prediction and highest-probability class are consistent."
            )

        # Show transaction input
        with st.expander("View Selected Transaction Features"):
            st.dataframe(
                transaction,
                use_container_width=True
            )

    except Exception as error:
        st.error("Prediction failed.")
        st.exception(error)

# ==================================================
# 8. FOOTER
# ==================================================

st.divider()

st.caption(
    "AI-Based Fraud Detection System | Machine Learning Project. "
    "Model predictions are estimates and are not definitive proof of fraud."
)
