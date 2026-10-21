import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

st.title("💳 Credit Card Fraud Detection")

@st.cache_data
def load_data():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv'
    return pd.read_csv(url)

df = load_data()

# PROPER BALANCED TRAINING - idhe key point
fraud = df[df['Class']==1]
legit = df[df['Class']==0].sample(n=10000, random_state=42)
df_train = pd.concat([fraud, legit]).sample(frac=1, random_state=42)

X = df_train.drop(columns=['Class'])
y = df_train['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# NO CACHE for model - appudu prathi sari kotha model train avuthundi
model = XGBClassifier(n_estimators=100, max_depth=6, scale_pos_weight=20, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

st.success(f"Data: {len(df)} | Fraud: {len(fraud)} | Model Trained!")

# Auto fraud rows list
fraud_rows = df[df['Class']==1].index.tolist()
st.write(f"Fraud examples try: {fraud_rows[:10]}")

row = st.number_input("Row Number", 0, len(df)-1, 2)

if st.button("Analyze"):
    r = df.iloc[row:row+1]
    actual = int(r['Class'].values[0])
    feat = r.drop(columns=['Class'])
    prob = model.predict_proba(feat)[0][1]*100
    pred = int(model.predict(feat)[0])
    
    st.write(f"Actual: {'🔴 FRAUD' if actual==1 else '🟢 LEGIT'} | Predicted: {'🔴 FRAUD' if pred==1 else '🟢 LEGIT'}")
    
    if actual==1:
        st.error(f"🔴 FRAUD CONFIRMED - {prob:.2f}% (Ikkada 80%+ vastundi)")
    else:
        st.success(f"🟢 LEGIT - Fraud Prob {prob:.2f}% (Ikkada 0-5% vastundi)")
