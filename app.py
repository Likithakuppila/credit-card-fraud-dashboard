import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection System")
st.write("Traditional vs Trending Models Comparison")

@st.cache_data
def load_data():
    # FIXED URL - Streamlit lo work ayyedi idhe
    url = 'https://raw.githubusercontent.com/mlg-ulb/creditcardfraud/master/creditcard.csv'
    df = pd.read_csv(url)
    return df

df = load_data()

X = df.drop(columns=['Class'])
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

st.sidebar.header("Model Settings")
model_choice = st.sidebar.selectbox("Select Model", ("Random Forest", "XGBoost"))

@st.cache_resource
def get_model(choice, X_train, y_train):
    if choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=20, class_weight='balanced', max_depth=10, random_state=42)
    else:
        model = XGBClassifier(scale_pos_weight=577, n_estimators=50, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    return model

with st.spinner('Training... 1 min'):
    model = get_model(model_choice, X_train, y_train)

st.success("Model Trained!")

st.header("Test a Transaction")
row_number = st.number_input("Enter Row Number", min_value=0, max_value=284806, value=406)

if st.button("Analyze Transaction"):
    transaction = df.iloc[row_number:row_number+1]
    actual_class = transaction['Class'].values[0]
    transaction_features = transaction.drop(columns=['Class'])
    prediction = model.predict(transaction_features)[0]
    fraud_prob = model.predict_proba(transaction_features)[0][1] * 100
    
    st.write(f"**Actual:** {'FRAUD' if actual_class==1 else 'LEGITIMATE'}")
    st.write(f"**Predicted:** {'FRAUD' if prediction==1 else 'LEGITIMATE'}")
    
    if prediction == 1:
        st.error(f"🔴 FRAUD TRANSACTION - Fraud Probability: {fraud_prob:.2f}%")
    else:
        st.success(f"🟢 LEGITIMATE TRANSACTION - Fraud Probability: {fraud_prob:.4f}%")
