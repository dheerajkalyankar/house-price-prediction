import os
import pandas as pd
from sklearn.model_selection import train_test_split


class DataIngestion:
    def __init__(self):
        """
        Initializes paths for raw data and ingestion artifacts
        """
        self.raw_data_path = os.path.join("data", "raw", "train.csv")
        self.ingested_dir = os.path.join("artifacts", "data_ingestion")

        self.train_path = os.path.join(self.ingested_dir, "train.csv")
        self.val_path = os.path.join(self.ingested_dir, "val.csv")

    def initiate_data_ingestion(self):
        """
        Reads raw data, performs train-validation split,
        and saves the processed data into artifacts folder
        """

        # 1. Create artifacts directory
        os.makedirs(self.ingested_dir, exist_ok=True)

        # 2. Read raw dataset
        df = pd.read_csv(self.raw_data_path)

        # 3. Train-validation split (ONLY from labeled data)
        train_df, val_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42
        )

        # 4. Save ingested datasets
        train_df.to_csv(self.train_path, index=False)
        val_df.to_csv(self.val_path, index=False)

        # 5. Logging (simple & useful)
        print("✅ Data Ingestion Completed")
        print(f"Train data shape: {train_df.shape}")
        print(f"Validation data shape: {val_df.shape}")

        return self.train_path, self.val_path
