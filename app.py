import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection")

@st.cache_data
def load_data():
    # 1st try - jsdelivr link (Streamlit lo work ayye link)
    try:
        url = 'https://cdn.jsdelivr.net/gh/mlg-ulb/creditcardfraud@master/creditcard.csv'
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.warning(f"Online data load fail ayyindi: {e}")
        # Fallback - local file unte
        try:
            df = pd.read_csv('creditcard.csv')
            return df
        except:
            st.error("Data load avvaledu. GitHub repo lo creditcard.csv pettandi.")
            st.stop()

df = load_data()

# Speed kosam data ni chinna chesam - App crash avvakunda
if len(df) > 30000:
    fraud = df[df['Class']==1]
    legit = df[df['Class']==0].sample(n=20000, random_state=42)
    df = pd.concat([fraud, legit]).sample(frac=1, random_state=42)

st.success(f"Data Ready: {len(df)} rows")

X = df.drop(columns=['Class'])
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model_choice = st.sidebar.selectbox("Model", ("Random Forest", "XGBoost"))

@st.cache_resource
def train(choice, X_train, y_train):
    if choice == "Random Forest":
        m = RandomForestClassifier(n_estimators=10, class_weight='balanced', random_state=42)
    else:
        m = XGBClassifier(n_estimators=10, scale_pos_weight=40, eval_metric='logloss', random_state=42)
    m.fit(X_train, y_train)
    return m

model = train(model_choice, X_train, y_train)

row = st.number_input("Row Number", 0, len(df)-1, 100)
if st.button("Analyze"):
    r = df.iloc[row:row+1]
    actual = r['Class'].values[0]
    feat = r.drop(columns=['Class'])
    pred = model.predict(feat)[0]
    prob = model.predict_proba(feat)[0][1]*100
    st.write(f"Actual: {'FRAUD' if actual==1 else 'LEGIT'} | Predicted: {'FRAUD' if pred==1 else 'LEGIT'}")
    if pred==1:
        st.error(f"🔴 FRAUD - {prob:.2f}%")
    else:
        st.success(f"🟢 LEGITIMATE - {prob:.4f}%")
