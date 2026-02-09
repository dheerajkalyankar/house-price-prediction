import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -----------------------------
# Load artifacts
# -----------------------------
BASE_DIR = "artifacts"

PREPROCESSOR_PATH = os.path.join(
    BASE_DIR, "data_transformation", "preprocessor.pkl"
)
MODEL_PATH = os.path.join(
    BASE_DIR, "model_trainer", "model.pkl"
)
TOP_INDICES_PATH = os.path.join(
    BASE_DIR, "model_trainer", "top_feature_indices.pkl"
)
FEATURE_NAMES_PATH = os.path.join(
    BASE_DIR, "data_transformation", "feature_names.pkl"
)

preprocessor = joblib.load(PREPROCESSOR_PATH)
model = joblib.load(MODEL_PATH)
top_indices = joblib.load(TOP_INDICES_PATH)
feature_names = joblib.load(FEATURE_NAMES_PATH)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🏠 House Price Prediction")

st.markdown("Enter house details:")

# ---- numeric inputs ----
overall_qual = st.slider("Overall Quality", 1, 10, 5)
gr_liv_area = st.number_input("Above Ground Living Area", value=1500)
total_bsmt_sf = st.number_input("Total Basement Area", value=800)
garage_cars = st.slider("Garage Cars", 0, 4, 2)
lot_area = st.number_input("Lot Area", value=7000)
year_built = st.slider("Year Built", 1900, 2024, 2005)
tot_rms = st.slider("Total Rooms Above Ground", 2, 15, 6)
full_bath = st.slider("Full Bathrooms", 0, 4, 2)

# ---- categorical inputs ----
bsmt_qual = st.selectbox("Basement Quality", ["Ex", "Gd", "TA", "Fa", "Po"])
kitchen_qual = st.selectbox("Kitchen Quality", ["Ex", "Gd", "TA", "Fa"])
neighborhood = st.selectbox(
    "Neighborhood",
    ["NoRidge", "Edwards", "NridgHt", "CollgCr", "OldTown"]
)

# -----------------------------
# Build FULL input dataframe
# -----------------------------
input_dict = {col: np.nan for col in feature_names}

input_dict.update({
    "OverallQual": overall_qual,
    "GrLivArea": gr_liv_area,
    "TotalBsmtSF": total_bsmt_sf,
    "GarageCars": garage_cars,
    "LotArea": lot_area,
    "YearBuilt": year_built,
    "TotRmsAbvGrd": tot_rms,
    "FullBath": full_bath,
    "BsmtQual": bsmt_qual,
    "KitchenQual": kitchen_qual,
    "Neighborhood": neighborhood
})

input_df = pd.DataFrame([input_dict])

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Price"):
    transformed = preprocessor.transform(input_df)

    # 🔥 THIS IS THE FIX 🔥
    transformed_top = transformed[:, top_indices]

    prediction = model.predict(transformed_top)[0]

    st.success(f"🏷️ Estimated House Price: ₹ {prediction:,.2f}")
