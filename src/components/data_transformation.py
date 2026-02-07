import os
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

import joblib


class DataTransformation:
    def __init__(self):
        self.ingestion_dir = os.path.join("artifacts", "data_ingestion")
        self.transformation_dir = os.path.join("artifacts", "data_transformation")

        self.train_path = os.path.join(self.ingestion_dir, "train.csv")
        self.val_path = os.path.join(self.ingestion_dir, "val.csv")

        self.preprocessor_path = os.path.join(self.transformation_dir, "preprocessor.pkl")

    def get_preprocessor(self, df):
        """
        Creates preprocessing pipeline for numerical and categorical features
        """

        # Separate feature types
        numerical_features = df.select_dtypes(include=["int64", "float64"]).columns
        categorical_features = df.select_dtypes(include=["object"]).columns

        # Remove target if present
        numerical_features = numerical_features.drop("SalePrice")

        # Numerical pipeline
        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]
        )

        # Categorical pipeline
        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]
        )

        # Column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, numerical_features),
                ("cat", cat_pipeline, categorical_features)
            ]
        )

        return preprocessor

    def initiate_data_transformation(self):
        # Create artifacts directory
        os.makedirs(self.transformation_dir, exist_ok=True)

        # Read ingested data
        train_df = pd.read_csv(self.train_path)
        val_df = pd.read_csv(self.val_path)

        # Split features and target
        target_column = "SalePrice"

        X_train = train_df.drop(columns=[target_column])
        y_train = train_df[target_column]

        X_val = val_df.drop(columns=[target_column])
        y_val = val_df[target_column]

        import joblib
        joblib.dump(
            X_train.columns.tolist(),
            "artifacts/data_transformation/feature_names.pkl"
        )



        # Create preprocessor
        preprocessor = self.get_preprocessor(train_df)

        # Fit ONLY on training data
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_val_transformed = preprocessor.transform(X_val)

        # Combine features and target
        train_arr = np.c_[X_train_transformed, y_train]
        val_arr = np.c_[X_val_transformed, y_val]

        # Save artifacts
        joblib.dump(preprocessor, self.preprocessor_path)

        np.save(os.path.join(self.transformation_dir, "train.npy"), train_arr)
        np.save(os.path.join(self.transformation_dir, "val.npy"), val_arr)

        print("✅ Data Transformation Completed")
        print(f"Transformed train shape: {train_arr.shape}")
        print(f"Transformed val shape: {val_arr.shape}")

        return (
            os.path.join(self.transformation_dir, "train.npy"),
            os.path.join(self.transformation_dir, "val.npy"),
            self.preprocessor_path
        )
