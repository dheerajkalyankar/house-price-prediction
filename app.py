from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np


# Load artifacts
preprocessor = joblib.load(
    "artifacts/data_transformation/preprocessor.pkl"
)
model = joblib.load(
    "artifacts/model_trainer/model.pkl"
)

app = FastAPI(title="House Price Prediction API")

# Input schema
class HouseFeatures(BaseModel):
    data: dict   # key-value pairs of features


@app.get("/")
def home():
    return {"message": "House Price Prediction API is running 🚀"}


feature_names = joblib.load(
    "artifacts/data_transformation/feature_names.pkl"
)

@app.post("/predict")
def predict_price(features: HouseFeatures):
    input_df = pd.DataFrame([features.data])

    # Align columns
    input_df = input_df.reindex(columns=feature_names, fill_value=np.nan)

    transformed = preprocessor.transform(input_df)
    prediction = model.predict(transformed)[0]

    return {
        "predicted_price": round(float(prediction), 2)
    }
