from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainingPipeline:
    def run_pipeline(self):
        print("🚀 Starting Training Pipeline")

        # 1. Data Ingestion
        ingestion = DataIngestion()
        train_path, val_path = ingestion.initiate_data_ingestion()

        # 2. Data Transformation
        transformation = DataTransformation()
        train_arr, val_arr, preprocessor_path = (
            transformation.initiate_data_transformation()
        )

        # 3. Model Training
        trainer = ModelTrainer()
        model_path = trainer.initiate_model_training()


        print("✅ Training Pipeline Completed Successfully")
        return model_path


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()
