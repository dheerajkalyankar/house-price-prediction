import streamlit as st
import requests

st.set_page_config(page_title="House Price Prediction", layout="centered")

st.title("🏠 House Price Prediction App")
st.write("Enter house details to predict the price")

# Input fields
OverallQual = st.slider("Overall Quality (1–10)", 1, 10, 7)
GrLivArea = st.number_input("Above Ground Living Area (sq ft)", value=1710)
GarageCars = st.slider("Garage Cars", 0, 5, 2)
TotalBsmtSF = st.number_input("Total Basement Area (sq ft)", value=856)
YearBuilt = st.number_input("Year Built", value=2003)

# FastAPI URL (your deployed API)
API_URL = "https://house-price-prediction-glff.onrender.com/predict"

if st.button("Predict Price 💰"):
    payload = {
        "data": {
            "OverallQual": OverallQual,
            "GrLivArea": GrLivArea,
            "GarageCars": GarageCars,
            "TotalBsmtSF": TotalBsmtSF,
            "YearBuilt": YearBuilt
        }
    }

    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        prediction = response.json()["predicted_price"]
        st.success(f"🏷️ Estimated House Price: ₹ {prediction:,.2f}")
    else:
        st.error("❌ Error predicting price")
