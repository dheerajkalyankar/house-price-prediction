import os
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

class ModelTrainer:
    def __init__(self):
        self.transformation_dir = "artifacts/data_transformation"
        self.trainer_dir = "artifacts/model_trainer"

        self.train_arr_path = os.path.join(self.transformation_dir, "train.npy")
        self.val_arr_path = os.path.join(self.transformation_dir, "val.npy")
        self.preprocessor_path = os.path.join(self.transformation_dir, "preprocessor.pkl")

        self.model_path = os.path.join(self.trainer_dir, "model.pkl")
        self.top_features_path = os.path.join(
            self.transformation_dir, "top_features.pkl"
        )
        self.top_indices_path = os.path.join(
        self.trainer_dir, "top_feature_indices.pkl"
)


        os.makedirs(self.trainer_dir, exist_ok=True)

    def initiate_model_training(self):
        # -----------------------------
        # Load transformed data
        # -----------------------------
        train_arr = np.load(self.train_arr_path)
        val_arr = np.load(self.val_arr_path)

        X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
        X_val, y_val = val_arr[:, :-1], val_arr[:, -1]

        # -----------------------------
        # Train initial model
        # -----------------------------
        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        r2 = r2_score(y_val, y_pred)

        print(f"✅ Initial Model R2 Score: {r2:.4f}")

        # -----------------------------
        # Feature Importance
        # -----------------------------
        importances = model.feature_importances_

        preprocessor = joblib.load(
        "artifacts/data_transformation/preprocessor.pkl"
        )

        feature_names = list(preprocessor.get_feature_names_out())



        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values(by="importance", ascending=False)

        # Save for analysis
        importance_df.to_csv(
            os.path.join(self.trainer_dir, "feature_importance.csv"),
            index=False
        )

        
        # -----------------------------
        # Select TOP N features
        # -----------------------------
        TOP_N = 12
        top_features = importance_df.head(TOP_N)["feature"].tolist()

        print("🔥 Top features selected:")
        for f in top_features:
                   print(f" - {f}")

        # -----------------------------
        # Get indices of top features
        # -----------------------------
        top_indices = [
            feature_names.index(f) for f in top_features
        ]

        # -----------------------------
        # Save artifacts
        # -----------------------------
        joblib.dump(top_features, self.top_features_path)
        joblib.dump(top_indices, self.top_indices_path)

        # -----------------------------
        # Retrain using ONLY top features
        # -----------------------------
        top_indices = [
            feature_names.index(f) for f in top_features
        ]

        X_train_top = X_train[:, top_indices]
        X_val_top = X_val[:, top_indices]

        final_model = RandomForestRegressor(
            n_estimators=300,
            random_state=42
        )

        final_model.fit(X_train_top, y_train)

        final_pred = final_model.predict(X_val_top)
        final_r2 = r2_score(y_val, final_pred)

        print(f"🚀 Final Model R2 (Top {TOP_N} features): {final_r2:.4f}")

        # Save final model
        joblib.dump(final_model, self.model_path)

        return self.model_path, self.top_features_path
