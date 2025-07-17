# src/pipeline/train_pipeline.py
import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainingPipeline:
    def __init__(self):
        logging.info("Training Pipeline initialized")

    def start_training(self):
        """
        Complete end-to-end training pipeline
        """
        try:
            logging.info("=" * 60)
            logging.info("STARTING COMPLETE TRAINING PIPELINE")
            logging.info("=" * 60)

            # Stage 1: Data Ingestion
            logging.info("STAGE 1: Data Ingestion Started")
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data Ingestion Completed Successfully")
            logging.info(f"Train data saved at: {train_data_path}")
            logging.info(f"Test data saved at: {test_data_path}")
            logging.info("-" * 60)

            # Stage 2: Data Transformation
            logging.info("STAGE 2: Data Transformation Started")
            data_transformation = DataTransformation()
            train_arr, test_arr, preprocessor_path = data_transformation.initiate_transformation(
                train_data_path, test_data_path
            )
            logging.info(f"Data Transformation Completed Successfully")
            logging.info(f"Preprocessor saved at: {preprocessor_path}")
            logging.info(f"Training array shape: {train_arr.shape}")
            logging.info(f"Testing array shape: {test_arr.shape}")
            logging.info("-" * 60)

            # Stage 3: Model Training
            logging.info("STAGE 3: Model Training Started")
            model_trainer = ModelTrainer()
            accuracy_score = model_trainer.initiate_model_training(train_arr, test_arr)
            logging.info(f"Model Training Completed Successfully")
            logging.info(f"Best Model Accuracy: {accuracy_score:.4f}")
            logging.info("-" * 60)

            logging.info("=" * 60)
            logging.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            logging.info("=" * 60)
            logging.info("Pipeline Summary:")
            logging.info(f"✅ Data Ingestion: {train_data_path}, {test_data_path}")
            logging.info(f"✅ Data Transformation: {preprocessor_path}")
            logging.info(f"✅ Model Training: Best Accuracy = {accuracy_score:.4f}")
            logging.info(f"✅ Model saved at: artifacts/model.pkl")
            
            # Check if all artifacts exist
            self.validate_artifacts()
            
            return {
                "status": "success",
                "train_data_path": train_data_path,
                "test_data_path": test_data_path,
                "preprocessor_path": preprocessor_path,
                "model_accuracy": accuracy_score,
                "artifacts": {
                    "model": "artifacts/model.pkl",
                    "preprocessor": "artifacts/preprocessor.pkl",
                    "train_data": train_data_path,
                    "test_data": test_data_path
                }
            }

        except Exception as e:
            logging.error("Training Pipeline Failed!")
            raise CustomException(e, sys)

    def validate_artifacts(self):
        """
        Validate that all required artifacts are created
        """
        try:
            required_artifacts = [
                "artifacts/model.pkl",
                "artifacts/preprocessor.pkl",
                "artifacts/train.csv",
                "artifacts/test.csv",
                "artifacts/data.csv"
            ]
            
            missing_artifacts = []
            for artifact in required_artifacts:
                if not os.path.exists(artifact):
                    missing_artifacts.append(artifact)
            
            if missing_artifacts:
                logging.error(f"Missing artifacts: {missing_artifacts}")
                raise CustomException(f"Pipeline failed - Missing artifacts: {missing_artifacts}", sys)
            else:
                logging.info("✅ All artifacts validated successfully!")
                
        except Exception as e:
            raise CustomException(e, sys)

    def get_pipeline_status(self):
        """
        Check the current status of pipeline artifacts
        """
        try:
            artifacts_status = {}
            artifacts = {
                "model": "artifacts/model.pkl",
                "preprocessor": "artifacts/preprocessor.pkl", 
                "train_data": "artifacts/train.csv",
                "test_data": "artifacts/test.csv",
                "raw_data": "artifacts/data.csv"
            }
            
            for name, path in artifacts.items():
                artifacts_status[name] = {
                    "exists": os.path.exists(path),
                    "path": path,
                    "size": os.path.getsize(path) if os.path.exists(path) else 0
                }
            
            return artifacts_status
            
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        # Initialize and run training pipeline
        pipeline = TrainingPipeline()
        result = pipeline.start_training()
        
        print("\n" + "="*80)
        print("🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY! 🎉")
        print("="*80)
        print(f"📊 Model Accuracy: {result['model_accuracy']:.4f}")
        print(f"📁 Artifacts created in: ./artifacts/")
        print(f"🔗 MLflow UI: http://localhost:5000")
        print("="*80)
        print("\n🚀 Ready to run Flask app: python app.py")
        
    except Exception as e:
        print(f"\n❌ Training Pipeline Failed: {str(e)}")
        sys.exit(1)
