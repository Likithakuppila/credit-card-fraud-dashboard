import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Fraud Detection Dashboard",
    page_icon="🔐",
    layout="wide"
)

st.title("🔐 AI-Based Fraud Detection Dashboard")
st.write(
    "Analyze digital payment transactions and identify "
    "potential fraudulent transactions using Machine Learning."
)

st.divider()

# --------------------------------------------------
# LOAD AVAILABLE MODELS
# --------------------------------------------------

available_models = {
    "Random Forest": "random_forest_model.pkl",
    "XGBoost": "xgboost_model.pkl"
}

available_models = {
    name: path
    for name, path in available_models.items()
    if os.path.exists(path)
}

if not available_models:
    st.error(
        "No trained models found. Please upload "
        "random_forest_model.pkl and xgboost_model.pkl "
        "to your GitHub repository."
    )
    st.stop()

# --------------------------------------------------
# UPLOAD DATASET
# --------------------------------------------------

st.header("📂 Upload Transaction Dataset")

uploaded_file = st.file_uploader(
    "Upload your creditcard.csv file",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload your creditcard.csv dataset to continue.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as error:
    st.error(f"Unable to read the CSV file: {error}")
    st.stop()

if df.empty:
    st.error("The uploaded CSV file is empty.")
    st.stop()

if "Class" not in df.columns:
    st.error(
        "The uploaded dataset must contain a 'Class' column. "
        "Expected: 0 = Legitimate, 1 = Fraud."
    )
    st.stop()

# --------------------------------------------------
# DATASET OVERVIEW
# --------------------------------------------------

st.header("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Transactions", f"{len(df):,}")
col2.metric("Legitimate Transactions", f"{(df['Class'] == 0).sum():,}")
col3.metric("Fraud Transactions", f"{(df['Class'] == 1).sum():,}")

with st.expander("View dataset preview"):
    st.dataframe(df.head(10), use_container_width=True)

with st.expander("View dataset information"):
    st.write("Dataset shape:", df.shape)
    st.write("Columns:", df.columns.tolist())
    st.write("Missing values:", int(df.isnull().sum().sum()))

# --------------------------------------------------
# VISUALIZATIONS
# --------------------------------------------------

st.header("📈 Transaction Analysis")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    class_counts = (
        df["Class"]
        .map({0: "Legitimate", 1: "Fraud"})
        .fillna(df["Class"].astype(str))
        .value_counts()
        .reset_index()
    )

    class_counts.columns = ["Transaction Type", "Count"]

    fig = px.bar(
        class_counts,
        x="Transaction Type",
        y="Count",
        title="Legitimate vs Fraud Transactions",
        color="Transaction Type"
    )

    st.plotly_chart(fig, use_container_width=True)

with chart_col2:
    if "Amount" in df.columns:
        fig_amount = px.histogram(
            df,
            x="Amount",
            nbins=40,
            title="Transaction Amount Distribution"
        )

        st.plotly_chart(fig_amount, use_container_width=True)
    else:
        st.info("The dataset does not contain an Amount column.")

# --------------------------------------------------
# AI FRAUD PREDICTION
# --------------------------------------------------

st.divider()
st.header("🤖 AI Fraud Prediction")

st.write(
    "Select a trained model and a transaction to estimate "
    "whether the transaction may be fraudulent."
)

selected_model_name = st.selectbox(
    "Choose Machine Learning Model",
    list(available_models.keys())
)

max_row = len(df) - 1

row_number = st.number_input(
    "Select Transaction Row Number",
    min_value=0,
    max_value=max_row,
    value=0,
    step=1
)

# --------------------------------------------------
# RUN PREDICTION
# --------------------------------------------------

if st.button("🔍 Analyze Transaction", type="primary"):

    model_path = available_models[selected_model_name]

    try:
        # Load the selected model
        model = joblib.load(model_path)

        # Remove target column before prediction
        X = df.drop(columns=["Class"]).copy()

        # Select exactly one transaction
        transaction = X.iloc[[int(row_number)]].copy()

        # Use the same feature order used during training
        if hasattr(model, "feature_names_in_"):
            expected_features = list(model.feature_names_in_)

            missing_features = [
                feature
                for feature in expected_features
                if feature not in transaction.columns
            ]

            if missing_features:
                st.error(
                    "Dataset is missing required model features: "
                    + ", ".join(missing_features)
                )
                st.stop()

            transaction = transaction[expected_features]

        # Ensure transaction values are numeric
        transaction = transaction.apply(
            pd.to_numeric,
            errors="coerce"
        )

        if transaction.isnull().any().any():
            st.error(
                "The selected transaction contains missing or "
                "non-numeric feature values. Please check the dataset."
            )
            st.stop()

        # Get model class labels
        model_classes = list(model.classes_)

        # Predict transaction class
        prediction = int(model.predict(transaction)[0])

        # Get probabilities for this exact transaction
        probabilities = model.predict_proba(transaction)[0]

        # Find the probability belonging to class 1 (Fraud)
        fraud_class_index = next(
            (
                index
                for index, class_label in enumerate(model_classes)
                if int(class_label) == 1
            ),
            None
        )

        if fraud_class_index is None:
            st.error(
                "The selected model does not contain class 1 "
                "for fraud detection. Please check the trained model."
            )
            st.stop()

        fraud_probability = float(
            probabilities[fraud_class_index]
        )

        # --------------------------------------------------
        # DISPLAY RESULTS
        # --------------------------------------------------

        st.subheader("Prediction Results")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            if prediction == 1:
                st.error("🚨 Prediction: FRAUD")
            else:
                st.success("✅ Prediction: LEGITIMATE")

        with result_col2:
            st.metric(
                "Estimated Fraud Probability",
                f"{fraud_probability * 100:.6f}%"
            )

        # Probability progress bar
        st.write("Fraud Probability Indicator")

        st.progress(
            min(max(fraud_probability, 0.0), 1.0)
        )

        # Show transaction details
        st.subheader("Selected Transaction Details")

        st.write("Transaction Row:", int(row_number))
        st.write("Selected Model:", selected_model_name)

        st.dataframe(
            transaction,
            use_container_width=True
        )

        # --------------------------------------------------
        # DEBUG INFORMATION
        # --------------------------------------------------

        with st.expander("🔧 Prediction Debug Information"):

            st.write("Model file:", model_path)
            st.write("Model classes:", model_classes)
            st.write("Model prediction:", prediction)
            st.write(
                "Raw probabilities:",
                probabilities.tolist()
            )
            st.write(
                "Fraud class probability index:",
                fraud_class_index
            )
            st.write(
                "Fraud probability (0 to 1):",
                fraud_probability
            )
            st.write(
                "Fraud probability (percentage):",
                fraud_probability * 100
            )
            st.write(
                "Input transaction shape:",
                transaction.shape
            )

            # Compare predicted class with probability-based class
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
                    "The model's predict() result differs from "
                    "the class with the highest predict_proba() "
                    "value. Please share this debug output."
                )
            else:
                st.success(
                    "Prediction and probability class are consistent."
                )

    except Exception as error:
        st.error("Prediction failed.")
        st.exception(error)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "AI-Based Fraud Detection System | "
    "Machine Learning Project | "
    "Predictions are estimates and should not be treated "
    "as definitive proof of fraud."
)
