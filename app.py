import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection System")
st.write("Traditional vs Trending Models Comparison")

@st.cache_data
def load_data():
    # Nee old method ne - Try local file, lekapothe online
    try:
        df = pd.read_csv('creditcard.csv')
    except:
        df = pd.read_csv('https://raw.githubusercontent.com/mlg-ulb/creditcardfraud/master/creditcard.csv')
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
        # Nee code lo okka line add chesa anthe - class_weight='balanced'
        model = RandomForestClassifier(n_estimators=20, class_weight='balanced', random_state=42)
    else:
        # Nee code lo okka line add chesa anthe - scale_pos_weight=577
        model = XGBClassifier(n_estimators=20, scale_pos_weight=577, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    return model

model = get_model(model_choice, X_train, y_train)
st.success("Model Ready!")

st.header("Test a Transaction")
row_number = st.number_input("Enter Row Number", min_value=0, max_value=len(df)-1, value=406)

if st.button("Analyze"):
    row = df.iloc[row_number:row_number+1]
    actual = row['Class'].values[0]
    prob = model.predict_proba(row.drop(columns=['Class']))[0][1]*100
    pred = model.predict(row.drop(columns=['Class']))[0]
    
    st.write(f"Actual: {'FRAUD' if actual==1 else 'LEGITIMATE'}")
    st.write(f"Predicted: {'FRAUD' if pred==1 else 'LEGITIMATE'}")
    
    if pred==1:
        st.error(f"🔴 FRAUD - {prob:.2f}%")
    else:
        st.success(f"🟢 LEGITIMATE - Fraud Prob {prob:.4f}%")
