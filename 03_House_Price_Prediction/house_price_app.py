
# house_price_app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")
st.title("🏠 House Price Predictor")

@st.cache_resource
def load_model():
    model = joblib.load("house_price_model.pkl")
    features = joblib.load("feature_names.pkl")
    return model, features

model, feature_names = load_model()

st.sidebar.header("House Features")

# Input sliders for key features
inputs = {}
for feature in feature_names:
    inputs[feature] = st.sidebar.slider(feature, 0.0, 10.0, 5.0, step=0.1)

if st.sidebar.button("Predict Price"):
    input_df = pd.DataFrame([inputs])
    log_pred = model.predict(input_df)[0]
    price = np.exp(log_pred)

    st.metric("Predicted Sale Price", f"${price:,.0f}")

    # Feature importance chart
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh([feature_names[i] for i in top_idx][::-1], importances[top_idx][::-1])
    ax.set_title("Top Feature Importances")
    st.pyplot(fig)
