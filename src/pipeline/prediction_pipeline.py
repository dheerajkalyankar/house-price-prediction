import os
import pandas as pd
import numpy as np
import joblib


class PredictionPipeline:
    def __init__(self):
        self.kaggle_test_path = os.path.join("data", "raw", "kaggle_test.csv")

        self.preprocessor_path = os.path.join(
            "artifacts", "data_transformation", "preprocessor.pkl"
        )
        self.model_path = os.path.join(
            "artifacts", "model_trainer", "model.pkl"
        )

    def run_prediction(self):
        # Load data
        test_df = pd.read_csv(self.kaggle_test_path)

        # Load artifacts
        preprocessor = joblib.load(self.preprocessor_path)
        model = joblib.load(self.model_path)

        # Transform test data
        X_test_transformed = preprocessor.transform(test_df)

        # Predict
        predictions = model.predict(X_test_transformed)

        # Create submission
        submission = pd.DataFrame({
            "Id": test_df["Id"],
            "SalePrice": predictions
        })

        submission.to_csv("submission.csv", index=False)
        print("✅ submission.csv created successfully")


if __name__ == "__main__":
    predictor = PredictionPipeline()
    predictor.run_prediction()
