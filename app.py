import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection System")
st.write("Traditional vs Trending Models Comparison - Light Version")

@st.cache_data
def load_data():
    # 150MB motham kakunda, konchem data ne fast ga load cheyadaniki
    url = 'https://raw.githubusercontent.com/mlg-ulb/creditcardfraud/master/creditcard.csv'
    # nrows lekapothe Streamlit crash avuthundi, anduke manam full data ni sample chestam
    df_full = pd.read_csv(url)
    
    # Speed kosam: Fraud anni + Legit 20000 matrame teeskuntam
    fraud = df_full[df_full['Class'] == 1]
    legit = df_full[df_full['Class'] == 0].sample(n=20000, random_state=42)
    df = pd.concat([fraud, legit])
    return df.sample(frac=1, random_state=42) # shuffle

df = load_data()
st.success(f"Data Loaded: {df.shape[0]} transactions (Fraud + Sample Legit)")

X = df.drop(columns=['Class'])
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

st.sidebar.header("Model Settings")
model_choice = st.sidebar.selectbox("Select Model", ("Random Forest", "XGBoost"))

@st.cache_resource
def get_model(choice, X_train, y_train):
    if choice == "Random Forest":
        return RandomForestClassifier(n_estimators=20, class_weight='balanced', max_depth=10, random_state=42).fit(X_train, y_train)
    else:
        return XGBClassifier(scale_pos_weight=40, n_estimators=50, eval_metric='logloss', random_state=42).fit(X_train, y_train)

with st.spinner('Training... 30 seconds...'):
    model = get_model(model_choice, X_train, y_train)

st.header("Test a Transaction")
# Sample data lo row numbers maarayi, anduke 0-20491 varaku matrame
row_number = st.number_input(f"Enter Row Number (0 to {len(df)-1})", min_value=0, max_value=len(df)-1, value=100)

if st.button("Analyze Transaction"):
    transaction = df.iloc[row_number:row_number+1]
    actual_class = transaction['Class'].values[0]
    transaction_features = transaction.drop(columns=['Class'])
    prediction = model.predict(transaction_features)[0]
    fraud_prob = model.predict_proba(transaction_features)[0][1] * 100
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Actual:** {'🔴 FRAUD' if actual_class==1 else '🟢 LEGITIMATE'}")
        st.write(f"**Predicted:** {'🔴 FRAUD' if prediction==1 else '🟢 LEGITIMATE'}")
    with col2:
        if prediction == 1:
            st.error(f"🔴 FRAUD - {fraud_prob:.2f}%")
        else:
            st.success(f"🟢 LEGITIMATE - Fraud Prob: {fraud_prob:.4f}%")
