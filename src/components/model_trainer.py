import os
import numpy as np
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_squared_error
import numpy as np


class ModelTrainer:
    def __init__(self):
        self.transformation_dir = os.path.join("artifacts", "data_transformation")
        self.model_dir = os.path.join("artifacts", "model_trainer")

        self.train_arr_path = os.path.join(self.transformation_dir, "train.npy")
        self.val_arr_path = os.path.join(self.transformation_dir, "val.npy")

        self.model_path = os.path.join(self.model_dir, "model.pkl")

    def evaluate_model(self, y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        return rmse, r2

    def initiate_model_trainer(self):
        # Create artifacts directory
        os.makedirs(self.model_dir, exist_ok=True)

        # Load transformed data
        train_arr = np.load(self.train_arr_path)
        val_arr = np.load(self.val_arr_path)

        # Split features and target
        X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
        X_val, y_val = val_arr[:, :-1], val_arr[:, -1]

        # Models to try
        models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            )
        }

        best_model = None
        best_rmse = float("inf")

        # Train & evaluate
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            rmse, r2 = self.evaluate_model(y_val, y_pred)

            print(f"{name} → RMSE: {rmse:.2f}, R2: {r2:.3f}")

            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model

        # Save best model
        joblib.dump(best_model, self.model_path)

        print("✅ Model training completed")
        print(f"Best model saved at: {self.model_path}")

        return self.model_path
