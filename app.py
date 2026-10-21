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
    url = 'https://github.com/mlg-ulb/creditcardfraud/raw/master/creditcard.csv'
    df = pd.read_csv(url)
    return df

df = load_data()
st.write(f"Dataset Loaded: {df.shape[0]} rows")

X = df.drop(columns=['Class'])
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

st.sidebar.header("Model Settings")
model_choice = st.sidebar.selectbox("Select Model", ("Random Forest", "XGBoost"))

# Training ni cache chestham - malla malla train avvakunda
@st.cache_resource
def get_model(choice, X_train, y_train):
    if choice == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=20,  # 100 nundi 20 ki taggincham - speed kosam
            class_weight='balanced',
            max_depth=10,
            random_state=42
        )
    else:
        model = XGBClassifier(
            scale_pos_weight=577,
            n_estimators=50, # Speed kosam
            eval_metric='logloss',
            random_state=42
        )
    model.fit(X_train, y_train)
    return model

with st.spinner('Training Model... First time 1-2 mins paduthundi'):
    model = get_model(model_choice, X_train, y_train)

st.success("Model Trained Successfully!")

st.header("Test a Transaction")
row_number = st.number_input("Enter Row Number (0 to 284806)", min_value=0, max_value=284806, value=406)

if st.button("Analyze Transaction"):
    transaction = df.iloc[row_number:row_number+1]
    actual_class = transaction['Class'].values[0]
    transaction_features = transaction.drop(columns=['Class'])
    
    prediction = model.predict(transaction_features)[0]
    probability = model.predict_proba(transaction_features)[0]
    fraud_prob = probability[1] * 100
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Actual:** {'🔴 FRAUD' if actual_class==1 else '🟢 LEGITIMATE'}")
        st.write(f"**Predicted:** {'🔴 FRAUD' if prediction==1 else '🟢 LEGITIMATE'}")
    
    with col2:
        if prediction == 1:
            st.error(f"🔴 FRAUD TRANSACTION")
            st.error(f"**Fraud Probability: {fraud_prob:.2f}%**")
        else:
            st.success(f"🟢 LEGITIMATE TRANSACTION")
            st.success(f"**Fraud Probability: {fraud_prob:.4f}%**")
