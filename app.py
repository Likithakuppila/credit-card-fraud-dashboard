import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os

st.set_page_config(
    page_title="AI Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide"
)

st.title("💳 AI-Based Fraud Detection in Digital Payment Gateways")
st.subheader("Transaction Analysis and Machine Learning Predictions")

st.divider()

# ---------------- DATASET UPLOAD ----------------

st.header("📂 Upload Transaction Dataset")

uploaded_file = st.file_uploader(
    "Upload creditcard.csv",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload your creditcard.csv file.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as error:
    st.error(f"Unable to read CSV file: {error}")
    st.stop()

if df.empty:
    st.error("The uploaded dataset is empty.")
    st.stop()

if "Class" not in df.columns:
    st.error("The dataset must contain a 'Class' column.")
    st.stop()

# Convert Class column into numeric values
df["Class"] = pd.to_numeric(df["Class"], errors="coerce")

# Count valid classes
valid_df = df[df["Class"].isin([0, 1])].copy()

if valid_df.empty:
    st.error("No valid Class values (0 or 1) found in the dataset.")

    with st.expander("Dataset diagnostic information"):
        st.write(df["Class"].value_counts(dropna=False).to_dict())

    st.stop()

total_transactions = len(df)
legitimate_count = int((valid_df["Class"] == 0).sum())
fraud_count = int((valid_df["Class"] == 1).sum())

classified_count = legitimate_count + fraud_count

fraud_percentage = (
    fraud_count / classified_count * 100
    if classified_count > 0
    else 0
)

# ---------------- SUMMARY CARDS ----------------

st.header("📊 Transaction Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Transactions", f"{total_transactions:,}")
col2.metric("Legitimate Transactions", f"{legitimate_count:,}")
col3.metric("Fraud Transactions", f"{fraud_count:,}")
col4.metric("Fraud Percentage", f"{fraud_percentage:.2f}%")

with st.expander("Dataset diagnostic information"):
    st.write("Dataset shape:", df.shape)
    st.write("Class data type:", str(df["Class"].dtype))
    st.write(
        "Class values:",
        df["Class"].value_counts(dropna=False).to_dict()
    )
    st.write("Columns:", df.columns.tolist())

with st.expander("Preview dataset"):
    st.dataframe(df.head(10), use_container_width=True)

# ---------------- FRAUD VS LEGITIMATE CHART ----------------

st.divider()
st.header("📈 Transaction Analysis")

chart_df = pd.DataFrame({
    "Transaction Type": ["Legitimate", "Fraud"],
    "Count": [legitimate_count, fraud_count]
})

fig_class = px.bar(
    chart_df,
    x="Transaction Type",
    y="Count",
    color="Transaction Type",
    text="Count",
    title="Fraud vs Legitimate Transactions"
)

st.plotly_chart(fig_class, use_container_width=True)

# ---------------- AMOUNT DISTRIBUTION ----------------

if "Amount" in df.columns:

    fig_amount = px.histogram(
        valid_df,
        x="Amount",
        nbins=50,
        title="Transaction Amount Distribution"
    )

    st.plotly_chart(fig_amount, use_container_width=True)

    amount_df = valid_df.copy()

    amount_df["Transaction Type"] = amount_df["Class"].map({
        0: "Legitimate",
        1: "Fraud"
    })

    fig_box = px.box(
        amount_df,
        x="Transaction Type",
        y="Amount",
        color="Transaction Type",
        title="Transaction Amount by Class"
    )

    st.plotly_chart(fig_box, use_container_width=True)

# ---------------- MACHINE LEARNING MODELS ----------------

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
        "Model files not found. Upload random_forest_model.pkl "
        "and xgboost_model.pkl to your GitHub repository."
    )

else:

    selected_model_name = st.selectbox(
        "Select Machine Learning Model",
        list(available_models.keys())
    )

    row_number = st.number_input(
        "Select transaction row number (starts at 0)",
        min_value=0,
        max_value=len(df) - 1,
        value=0,
        step=1
    )

    if st.button("🔍 Analyze Transaction", type="primary"):

        try:
            model_path = available_models[selected_model_name]
            model = joblib.load(model_path)

            # Separate input features and target column
            X = df.drop(columns=["Class"]).copy()

            transaction = X.iloc[[int(row_number)]].copy()

            # Match the feature order used to train the model
            if hasattr(model, "feature_names_in_"):

                expected_features = list(model.feature_names_in_)

                missing_features = [
                    feature
                    for feature in expected_features
                    if feature not in transaction.columns
                ]

                if missing_features:
                    st.error(
                        "Required features are missing: "
                        + ", ".join(missing_features)
                    )
                    st.stop()

                transaction = transaction[expected_features]

            transaction = transaction.apply(
                pd.to_numeric,
                errors="coerce"
            )

            if transaction.isnull().any().any():
                st.error(
                    "Selected transaction contains missing or "
                    "non-numeric feature values."
                )
                st.stop()

            # Prediction and probability
            prediction = int(model.predict(transaction)[0])

            probabilities = model.predict_proba(transaction)[0]

            model_classes = list(model.classes_)

            fraud_indices = [
                i for i, label in enumerate(model_classes)
                if int(label) == 1
            ]

            if not fraud_indices:
                st.error("The selected model has no fraud class (1).")
                st.stop()

            fraud_index = fraud_indices[0]

            fraud_probability = float(probabilities[fraud_index])

            # ---------------- DISPLAY RESULTS ----------------

            st.subheader("Prediction Result")

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                if prediction == 1:
                    st.error("🚨 FRAUD DETECTED")
                else:
                    st.success("✅ LEGITIMATE TRANSACTION")

            with result_col2:
                st.metric(
                    "Estimated Fraud Probability",
                    f"{fraud_probability * 100:.6f}%"
                )

            st.write("Fraud Probability Indicator")
            st.progress(min(max(fraud_probability, 0.0), 1.0))

            st.subheader("Selected Transaction Details")
            st.write("Transaction row:", int(row_number))
            st.write("Selected model:", selected_model_name)

            st.dataframe(
                transaction,
                use_container_width=True
            )

            # ---------------- DEBUG INFORMATION ----------------

            with st.expander("🔧 Prediction Debug Information"):

                st.write("Model file:", model_path)
                st.write("Model classes:", model_classes)
                st.write("Prediction:", prediction)
                st.write("Raw probabilities:", probabilities.tolist())
                st.write("Fraud probability index:", fraud_index)
                st.write("Fraud probability:", fraud_probability)

                highest_probability_class = int(
                    model_classes[int(probabilities.argmax())]
                )

                st.write(
                    "Class with highest probability:",
                    highest_probability_class
                )

                if prediction == highest_probability_class:
                    st.success(
                        "Prediction and probability class are consistent."
                    )
                else:
                    st.warning(
                        "Prediction and highest-probability class differ. "
                        "Please share this debug output."
                    )

        except Exception as error:
            st.error("Unable to analyze this transaction.")
            st.exception(error)

# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "AI-Based Fraud Detection System | Machine Learning Project | "
    "Predictions are estimates and are not definitive proof of fraud."
)
