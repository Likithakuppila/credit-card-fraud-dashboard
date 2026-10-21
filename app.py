import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

st.title("💳 Credit Card Fraud Detection")

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

# Rendu models train - No cache
rf = RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42)
xgb = XGBClassifier(n_estimators=100, max_depth=6, scale_pos_weight=20, eval_metric='logloss', random_state=42)
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

st.success(f"Data: {len(df)} | Fraud: {len(fraud)} | Both Models Ready!")

# Test kosam buttons - List kanipinchadu, buttons matrame
colA, colB = st.columns(2)
with colA:
    if st.button("🎲 Test FRAUD"):
        st.session_state['row'] = int(df[df['Class']==1].sample(1).index[0])
with colB:
    if st.button("🎲 Test LEGIT"):
        st.session_state['row'] = int(df[df['Class']==0].sample(1).index[0])

row = st.number_input("Row Number", 0, len(df)-1, value=st.session_state.get('row', 541))

if st.button("Analyze Both Models"):
    r = df.iloc[row:row+1]
    actual = int(r['Class'].values[0])
    feat = r.drop(columns=['Class'])
    
    st.write(f"**Actual Data: {'🔴 FRAUD' if actual==1 else '🟢 LEGITIMATE'}**")
    
    c1, c2 = st.columns(2)
    with c1:
        prob_rf = rf.predict_proba(feat)[0][1]*100
        pred_rf = rf.predict(feat)[0]
        st.info(f"**Random Forest (Traditional)**\n\nPredicted: {'FRAUD' if pred_rf==1 else 'LEGIT'}\n\nProb: {prob_rf:.2f}%")
    with c2:
        prob_xgb = xgb.predict_proba(feat)[0][1]*100
        pred_xgb = xgb.predict(feat)[0]
        st.error(f"**XGBoost (Trending)**\n\nPredicted: {'FRAUD' if pred_xgb==1 else 'LEGIT'}\n\nProb: {prob_xgb:.2f}%")
    
    st.write("---")
    if prob_xgb > prob_rf and actual==1:
        st.success("✅ XGBoost is more confident for FRAUD! - Trending is Better")
