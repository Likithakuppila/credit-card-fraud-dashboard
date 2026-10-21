import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection Dashboard")
st.write("Interactive Transaction Analysis")

uploaded_file = st.file_uploader(
    "Upload your creditcard.csv dataset",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if not {"Class", "Amount"}.issubset(df.columns):
        st.error("Dataset must contain Class and Amount columns.")
        st.stop()

    df["Amount"] = pd.to_numeric(
        df["Amount"], errors="coerce"
    )
    df = df.dropna(subset=["Class", "Amount"])

    st.sidebar.header("Filters")

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
        df["Amount"].between(
            amount_range[0], amount_range[1]
        )
    ]

    if transaction_type == "Legitimate (0)":
        filtered = filtered[filtered["Class"] == 0]
    elif transaction_type == "Fraud (1)":
        filtered = filtered[filtered["Class"] == 1]

    total = len(filtered)
    fraud = int((filtered["Class"] == 1).sum())
    legitimate = int((filtered["Class"] == 0).sum())
    fraud_pct = fraud / total * 100 if total else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{total:,}")
    c2.metric("Fraud Transactions", f"{fraud:,}")
    c3.metric("Legitimate Transactions", f"{legitimate:,}")
    c4.metric("Fraud Percentage", f"{fraud_pct:.2f}%")

    if total:
        counts = (
            filtered["Class"]
            .map({0: "Legitimate", 1: "Fraud"})
            .value_counts()
            .rename_axis("Transaction Type")
            .reset_index(name="Count")
        )

        st.subheader("Fraud vs Legitimate Transactions")
        st.plotly_chart(
            px.bar(counts, x="Transaction Type", y="Count",
                   color="Transaction Type"),
            use_container_width=True
        )

        st.subheader("Transaction Amount Distribution")
        chart_data = filtered.sample(
            n=min(10000, total), random_state=42
        )
        st.plotly_chart(
            px.histogram(
                chart_data, x="Amount", color="Class",
                nbins=40
            ),
            use_container_width=True
        )

        st.subheader("Transaction Amount by Class")
        st.plotly_chart(
            px.box(chart_data, x="Class", y="Amount"),
            use_container_width=True
        )
    else:
        st.warning("No transactions match these filters.")

else:
    st.info("Upload your creditcard.csv file to begin.")
