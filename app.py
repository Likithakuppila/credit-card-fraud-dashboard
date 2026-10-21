```python
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os

# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="AI Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide"
)

st.title("💳 AI-Based Fraud Detection in Digital Payment Gateways")
st.subheader("Transaction Analysis and Machine Learning Predictions")

st.divider()

# ==================================================
# MODEL FILES
# ==================================================

MODEL_FILES = {
    "Random Forest": "random_forest_model.pkl",
    "XGBoost": "xgboost_model.pkl"
}

available_models = {
    name: path
    for name, path in MODEL_FILES.items()
    if os.path.exists(path)
}

# ==================================================
# DATASET UPLOAD
# ==================================================

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

# ==================================================
# VALIDATE DATASET
# ==================================================

if "Class" not in df.columns:
    st.error(
        "The dataset must contain a 'Class' column. "
        "Expected values: 0 = Legitimate, 1 = Fraud."
    )
    st.stop()

# Convert Class column to numeric values.
# Invalid values become missing rather than being counted as fraud.
df["Class"] = pd.to_numeric(df["Class"], errors="coerce")

# Remove rows without a valid target class for analysis.
valid_class_df = df[df["Class"].isin([0, 1])].copy()

if valid_class_df.empty:
    st.error(
        "No valid Class values (0 or 1) were found. "
        "Please check the uploaded dataset."
    )

    with st.expander("Inspect Class column"):
        st.write(df["Class"].value_counts(dropna=False).to_dict())

    st.stop()

# ==================================================
# DATASET OVERVIEW
# ==================================================

st.header("📊 Transaction Overview")

total_transactions = len(df)

legitimate_count = int(
    (valid_class_df["Class"] == 0).sum()
)

fraud_count = int(
    (valid_class_df["Class"] == 1).sum()
)

classified_count = legitimate_count + fraud_count

fraud_percentage = (
    fraud_count / classified_count * 100
    if classified_count > 0
    else 0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Transactions",
    f"{total_transactions:,}"
)

col2.metric(
    "Legitimate Transactions",
    f"{legitimate_count:,}"
)

col3.metric(
    "Fraud Transactions",
    f"{fraud_count:,}"
)

col4.metric(
    "Fraud Percentage",
    f"{fraud_percentage:.2f}%"
)

with st.expander("🔎 Dataset diagnostic information"):
    st.write("Dataset shape:", df.shape)
    st.write("Class column type:", str(df["Class"].dtype))

    st.write(
        "Class value counts:",
        df["Class"].value_counts(dropna=False).to_dict()
    )

    st.write(
        "Rows with missing or invalid Class:",
        int((~df["Class"].isin([0, 1])).sum())
    )

    st.write("Column names:", df.columns.tolist())

# ==================================================
# DATASET PREVIEW
# ==================================================

with st.expander("👀 View Dataset Preview"):
    st.dataframe(
        df.head(10),
        use_container_width=True
    )

# ==================================================
# VISUALIZATION 1: FRAUD VS LEGITIMATE
# ==================================================

st.divider()
st.header("📈 Transaction Analysis")

chart_df = pd.DataFrame({
    "Transaction Type": [
        "Legitimate",
        "Fraud"
    ],
    "Count": [
        legitimate_count,
        fraud_count
    ]
})

fig_class = px.bar(
    chart_df,
    x="Transaction Type",
    y="Count",
    color="Transaction Type",
    title="Fraud vs Legitimate Transactions",
    text="Count"
)

fig_class.update_layout(
    xaxis_title="Transaction Type",
    yaxis_title="Number of Transactions"
)

st.plotly_chart(
    fig_class,
    use_container_width=True
)

# ==================================================
# VISUALIZATION 2: TRANSACTION AMOUNT DISTRIBUTION
# ==================================================

if "Amount" in df.columns:

    amount_df = valid_class_df.copy()

    fig_amount = px.histogram(
        amount_df,
        x="Amount",
        nbins=50,
        title="Transaction Amount Distribution"
    )

    st.plotly_chart(
        fig_amount,
        use_container_width=True
    )

    # ==================================================
    # VISUALIZATION 3: AMOUNT BY CLASS
    # ==================================================

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

    st.plotly_chart(
        fig_box,
        use_container_width=True
    )

else:
    st.warning(
        "The uploaded dataset does not contain an Amount column."
    )

# ==================================================
# AI FRAUD PREDICTION
# ==================================================

st.divider()
st.header("🤖 AI Fraud Prediction")

st.write(
    "Choose a machine learning model and transaction row "
    "to estimate the probability of fraud."
)

if not available_models:

    st.error(
        "No model files were found. Please upload "
        "random_forest_model.pkl and/or xgboost_model.pkl "
        "to your GitHub repository."
    )

else:

    selected_model_name = st.selectbox(
        "Select Machine Learning Model",
        list(available_models.keys())
    )

    max_row = len(df) - 1

    row_number = st.number_input(
        "Transaction row number (starting from 0)",
        min_value=0,
        max_value=max_row,
        value=0,
        step=1
    )

    if st.button("🔍 Analyze Transaction", type="primary"):

        try:

            # Load the selected model
            model_path = available_models[selected_model_name]

            model = joblib.load(model_path)

            # Prepare features by removing target column
            X = df.drop(columns=["Class"]).copy()

            transaction = X.iloc[
                [int(row_number)]
            ].copy()

            # Ensure feature order matches training
            if hasattr(model, "feature_names_in_"):

                expected_features = list(
                    model.feature_names_in_
                )

                missing_features = [
                    feature
                    for feature in expected_features
                    if feature not in transaction.columns
                ]

                if missing_features:
                    st.error(
                        "Missing required features: "
                        + ", ".join(missing_features)
                    )
                    st.stop()

                transaction = transaction[
                    expected_features
                ]

            # Convert input features to numeric
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

            # Get model classes
            model_classes = list(model.classes_)

            # Predict class and probabilities
            prediction = int(
                model.predict(transaction)[0]
            )

            probabilities = model.predict_proba(
                transaction
            )[0]

            # Locate class 1 (fraud)
            fraud_indices = [
                index
                for index, label in enumerate(model_classes)
                if int(label) == 1
            ]

            if not fraud_indices:
                st.error(
                    "The selected model does not have a fraud "
                    "class (1). Please check the trained model."
                )
                st.stop()

            fraud_index = fraud_indices[0]

            fraud_probability = float(
                probabilities[fraud_index]
            )

            # ==========================================
            # DISPLAY PREDICTION
            # ==========================================

            st.subheader("Prediction Result")

            result_col1, result_col2 = st.columns(2)

            with result_col1:

                if prediction == 1:
                    st.error("🚨 FRAUD DETECTED")

                elif prediction == 0:
                    st.success("✅ LEGITIMATE TRANSACTION")

                else:
                    st.warning(
                        f"Unexpected prediction class: {prediction}"
                    )

            with result_col2:

                st.metric(
                    "Estimated Fraud Probability",
                    f"{fraud_probability * 100:.6f}%"
                )

            st.write("Fraud Probability Indicator")

            st.progress(
                min(
                    max(fraud_probability, 0.0),
                    1.0
                )
            )

            # ==========================================
            # TRANSACTION DETAILS
            # ==========================================

            st.subheader("Selected Transaction Details")

            st.write(
                "Transaction row:",
                int(row_number)
            )

            st.write(
                "Selected model:",
                selected_model_name
            )

            st.dataframe(
                transaction,
                use_container_width=True
            )

            # ==========================================
            # DEBUG INFORMATION
            # ==========================================

            with st.expander(
                "🔧 Prediction Debug Information"
            ):

                st.write(
                    "Model file:",
                    model_path
                )

                st.write(
                    "Model classes:",
                    model_classes
                )

                st.write(
                    "Prediction:",
                    prediction
                )

                st.write(
                    "Raw probabilities:",
                    probabilities.tolist()
                )

                st.write(
                    "Fraud probability index:",
                    fraud_index
                )

                st.write(
                    "Fraud probability (0 to 1):",
                    fraud_probability
                )

                probability_prediction = int(
                    model_classes[
                        int(probabilities.argmax())
                    ]
                )

                st.write(
                    "Class with highest probability:",
                    probability_prediction
                )

                if prediction != probability_prediction:

                    st.warning(
                        "Prediction and highest-probability class "
                        "do not match. Please share this debug output."
                    )

                else:

                    st.success(
                        "Prediction and probability class are consistent."
                    )

        except Exception as error:

            st.error("Unable to analyze this transaction.")

            st.exception(error)

# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "AI-Based Fraud Detection System | "
    "Machine Learning Project | "
    "Predictions are estimates and should not be treated "
    "as definitive proof of fraud."
)
```
