import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import time

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

st.title("💳 Credit Card Fraud Detection System")
st.write("Traditional vs Trending Models Comparison")

# 1. DATA LOAD CHEYADAM
@st.cache_data
def load_data():
    df = pd.read_csv('creditcard.csv')  # GitHub lo creditcard.csv undali
    return df

df = load_data()

st.sidebar.header("Model Settings")
model_choice = st.sidebar.selectbox("Select Model", ("Random Forest", "XGBoost"))

# 2. DATA PREPARE CHEYADAM
X = df.drop(columns=['Class'])
y = df['Class']

# IMPORTANT: stratify=y pettam - Fraud, Legit ratio maintain cheyadaniki
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. MODEL TRAIN CHEYADAM - IKKADA FIX UNDI
if model_choice == "Random Forest":
    st.sidebar.write("**Traditional Model**")
    # FIX: class_weight='balanced' add chesam
    model = RandomForestClassifier(
        n_estimators=100, 
        class_weight='balanced',  # Fraud ki ekkuva importance isthundi
        random_state=42
    )
else:
    st.sidebar.write("**Trending Model**")
    # FIX: scale_pos_weight=577 add chesam
    # 577 = 284315 / 492 approx - Legit / Fraud ratio
    model = XGBClassifier(
        scale_pos_weight=577,  # Fraud ki ekkuva weight isthundi
        n_estimators=100,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )

# Model ni train cheyadam
with st.spinner('Training Model...'):
    model.fit(X_train, y_train)

st.success("Model Trained Successfully!")

# 4. PREDICTION PART
st.header("Test a Transaction")

row_number = st.number_input("Enter Row Number to Test", min_value=0, max_value=len(df)-1, value=406)

if st.button("Analyze Transaction"):
    # Row data teeskovadam
    transaction = df.iloc[row_number:row_number+1]
    actual_class = transaction['Class'].values[0]
    transaction_features = transaction.drop(columns=['Class'])
    
    # Prediction cheyadam
    prediction = model.predict(transaction_features)[0]
    probability = model.predict_proba(transaction_features)[0]
    
    fraud_prob = probability[1] * 100
    legit_prob = probability[0] * 100
    
    st.write("### Results:")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Actual Class:** {'FRAUD' if actual_class==1 else 'LEGITIMATE'}")
        st.write(f"**Model Prediction:** {'FRAUD' if prediction==1 else 'LEGITIMATE'}")
    
    with col2:
        if prediction == 1:
            st.error(f"🔴 FRAUD TRANSACTION")
            st.error(f"**Estimated Fraud Probability: {fraud_prob:.2f}%**")
        else:
            st.success(f"🟢 LEGITIMATE TRANSACTION")
            st.success(f"**Estimated Fraud Probability: {fraud_prob:.4f}%**")
    
    # Performance kuda chupistundi
    st.header("Model Performance on Test Data")
    y_pred = model.predict(X_test)
    st.text(classification_report(y_test, y_pred))
