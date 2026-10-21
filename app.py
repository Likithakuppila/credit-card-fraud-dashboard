import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

st.title("💳 Credit Card Fraud Detection System")
st.write("Traditional vs Trending Models Comparison")

# 1. DATA LOAD - FIXED WITH URL
@st.cache_data
def load_data():
    url = 'https://github.com/mlg-ulb/creditcardfraud/raw/master/creditcard.csv'
    df = pd.read_csv(url)
    return df

df = load_data()

st.sidebar.header("Model Settings")
model_choice = st.sidebar.selectbox("Select Model", ("Random Forest", "XGBoost"))

# 2. DATA PREPARE
X = df.drop(columns=['Class'])
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. MODEL TRAIN - FIXED WITH BALANCED WEIGHT
if model_choice == "Random Forest":
    st.sidebar.info("**Traditional Model - Random Forest**")
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42
    )
else:
    st.sidebar.info("**Trending Model - XGBoost**")
    model = XGBClassifier(
        scale_pos_weight=577,
        n_estimators=100,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )

with st.spinner('Training Model... Please wait'):
    model.fit(X_train, y_train)

st.success("Model Trained Successfully!")

# 4. PREDICTION PART
st.header("Test a Transaction")
row_number = st.number_input("Enter Row Number (0 to 284806)", min_value=0, max_value=284806, value=406)

if st.button("Analyze Transaction"):
    transaction = df.iloc[row_number:row_number+1]
    actual_class = transaction['Class'].values[0]
    transaction_features = transaction.drop(columns=['Class'])
    
    prediction = model.predict(transaction_features)[0]
    probability =
