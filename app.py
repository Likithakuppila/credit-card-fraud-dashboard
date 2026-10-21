import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

st.set_page_config(page_title="AI Based Fraud Detection", layout="wide")
st.title("🤖 AI Based Fraud Detection in Digital Payment Gateways")
st.subheader("Traditional vs Trending Model Comparison")
st.markdown("---")

@st.cache_data
def load_data():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv'
    return pd.read_csv(url)

df = load_data()
fraud = df[df['Class']==1]
legit = df[df['Class']==0].sample(n=10000, random_state=42)
df_train = pd.concat([fraud, legit]).sample(frac=1, random_state=42)

X = df_train.drop(columns=['Class'])
y = df_train['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42)
xgb = XGBClassifier(n_estimators=100, max_depth=6, scale_pos_weight=20, eval_metric='logloss', random_state=42)
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

st.success(f"✅ Digital Payment Data: {len(df)} | Fraud Cases: {len(fraud)} | AI Models Ready!")

cA, cB = st.columns(2)
with cA:
    if st.button("🎲 Test FRAUD Transaction"):
        st.session_state['row'] = int(df[df['Class']==1].sample(1).index[0])
with cB:
    if st.button("🎲 Test LEGITIMATE (Good) Transaction"):
        st.session_state['row'] = int(df[df['Class']==0].sample(1).index[0])

row = st.number_input("Enter Payment Gateway Transaction ID (0 to 284807)", 0, len(df)-1, value=st.session_state.get('row', 105178))

if st.button("🔍 Analyze with AI"):
    r = df.iloc[row:row+1]
    actual = int(r['Class'].values[0])
    feat = r.drop(columns=['Class'])
    
    st.subheader(f"Actual Payment: {'🔴 FRAUD' if actual==1 else '🟢 LEGITIMATE (Genuine Payment)'}")
    
    col1, col2 = st.columns(2)
    with col1:
        prob_rf = rf.predict_proba(feat)[0][1]*100
        st.info(f"**Traditional AI - Random Forest**\n\nPrediction: {'FRAUD' if rf.predict(feat)[0]==1 else 'LEGITIMATE'}\n\nRisk Score: {prob_rf:.2f}%")
    with col2:
        prob_xgb = xgb.predict_proba(feat)[0][1]*100
        st.warning(f"**Trending AI - XGBoost**\n\nPrediction: {'FRAUD' if xgb.predict(feat)[0]==1 else 'LEGITIMATE'}\n\nRisk Score: {prob_xgb:.2f}%")
    
    st.divider()
    if actual==1:
        st.error(f"🚨 ALERT: AI Detected Fraud in Digital Payment Gateway! XGBoost Confidence: {prob_xgb:.2f}%")
    else:
        st.success("✅ SAFE: This is a Genuine Payment in Digital Gateway!")
