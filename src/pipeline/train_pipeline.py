# src/pipeline/train_pipeline.py - FIXED VERSION
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
        Complete end-to-end training pipeline with robust error handling
        """
        try:
            logging.info("=" * 60)
            logging.info("STARTING COMPLETE TRAINING PIPELINE")
            logging.info("=" * 60)

            # Stage 1: Data Ingestion
            logging.info("STAGE 1: Data Ingestion Started")
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
            
            # Validate data ingestion outputs
            if not os.path.exists(train_data_path) or not os.path.exists(test_data_path):
                raise CustomException("Data ingestion failed - output files not created", sys)
            
            logging.info(f"✅ Data Ingestion Completed Successfully")
            logging.info(f"📄 Train data: {train_data_path} ({os.path.getsize(train_data_path):,} bytes)")
            logging.info(f"📄 Test data: {test_data_path} ({os.path.getsize(test_data_path):,} bytes)")
            logging.info("-" * 60)

            # Stage 2: Data Transformation
            logging.info("STAGE 2: Data Transformation Started")
            data_transformation = DataTransformation()
            train_arr, test_arr, preprocessor_path = data_transformation.initiate_transformation(
                train_data_path, test_data_path
            )
            
            # Validate data transformation outputs
            if not os.path.exists(preprocessor_path):
                raise CustomException("Data transformation failed - preprocessor not saved", sys)
            
            if train_arr is None or test_arr is None:
                raise CustomException("Data transformation failed - arrays not created", sys)
            
            logging.info(f"✅ Data Transformation Completed Successfully")
            logging.info(f"📄 Preprocessor: {preprocessor_path} ({os.path.getsize(preprocessor_path):,} bytes)")
            logging.info(f"📊 Training array shape: {train_arr.shape}")
            logging.info(f"📊 Testing array shape: {test_arr.shape}")
            logging.info("-" * 60)

            # Stage 3: Model Training
            logging.info("STAGE 3: Model Training Started")
            model_trainer = ModelTrainer()
            accuracy_score, model_report = model_trainer.initiate_model_training(train_arr, test_arr)
            
            # Validate model training outputs
            model_path = "artifacts/model.pkl"
            if not os.path.exists(model_path):
                raise CustomException("Model training failed - model not saved", sys)
            
            if accuracy_score is None or accuracy_score < 0.1:
                raise CustomException(f"Model training failed - poor accuracy: {accuracy_score}", sys)
            
            logging.info(f"✅ Model Training Completed Successfully")
            logging.info(f"📄 Model: {model_path} ({os.path.getsize(model_path):,} bytes)")
            logging.info(f"🎯 Best Model Accuracy: {accuracy_score:.4f}")
            logging.info("-" * 60)

            # Final validation of all artifacts
            logging.info("STAGE 4: Final Artifact Validation")
            self.validate_artifacts()
            
            logging.info("=" * 60)
            logging.info("🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            logging.info("=" * 60)
            
            # Print comprehensive summary
            summary = self.get_pipeline_summary()
            for key, value in summary.items():
                logging.info(f"✅ {key}: {value}")
            
            return {
                "status": "success",
                "train_data_path": train_data_path,
                "test_data_path": test_data_path,
                "preprocessor_path": preprocessor_path,
                "model_accuracy": accuracy_score,
                "model_report": model_report,
                "artifacts": self.get_artifact_paths(),
                "summary": summary
            }

        except Exception as e:
            logging.error("🚨 TRAINING PIPELINE FAILED!")
            logging.error(f"Error details: {str(e)}")
            
            # Try to provide helpful debugging information
            self.debug_pipeline_state()
            
            raise CustomException(e, sys)

    def validate_artifacts(self):
        """
        Validate that all required artifacts are created and accessible
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
            corrupted_artifacts = []
            
            for artifact in required_artifacts:
                if not os.path.exists(artifact):
                    missing_artifacts.append(artifact)
                else:
                    # Check if file is accessible and not corrupted
                    try:
                        size = os.path.getsize(artifact)
                        if size == 0:
                            corrupted_artifacts.append(f"{artifact} (empty file)")
                        else:
                            logging.info(f"✅ {artifact}: {size:,} bytes")
                    except Exception as e:
                        corrupted_artifacts.append(f"{artifact} (access error: {e})")
            
            # Report any issues
            if missing_artifacts:
                raise CustomException(f"Missing artifacts: {missing_artifacts}", sys)
            
            if corrupted_artifacts:
                raise CustomException(f"Corrupted artifacts: {corrupted_artifacts}", sys)
                
            logging.info("✅ All artifacts validated successfully!")
                
        except Exception as e:
            raise CustomException(e, sys)

    def get_pipeline_summary(self):
        """
        Get a comprehensive summary of the pipeline execution
        """
        try:
            artifacts = self.get_artifact_paths()
            summary = {}
            
            for name, path in artifacts.items():
                if os.path.exists(path):
                    size = os.path.getsize(path)
                    summary[f"{name}_status"] = f"Created ({size:,} bytes)"
                else:
                    summary[f"{name}_status"] = "❌ Missing"
            
            # Add timing and performance info
            summary["pipeline_status"] = "✅ Complete"
            summary["ready_for_inference"] = "✅ Yes"
            summary["mlflow_ui"] = "http://localhost:5000"
            summary["next_steps"] = "Run 'python app.py' to start web interface"
            
            return summary
            
        except Exception as e:
            logging.warning(f"Could not generate pipeline summary: {e}")
            return {"status": "Summary generation failed"}

    def get_artifact_paths(self):
        """
        Get dictionary of all artifact paths
        """
        return {
            "model": "artifacts/model.pkl",
            "preprocessor": "artifacts/preprocessor.pkl",
            "train_data": "artifacts/train.csv",
            "test_data": "artifacts/test.csv",
            "raw_data": "artifacts/data.csv"
        }

    def debug_pipeline_state(self):
        """
        Provide debugging information when pipeline fails
        """
        try:
            logging.error("🔍 DEBUGGING INFORMATION:")
            
            # Check directory structure
            if os.path.exists("artifacts"):
                files = os.listdir("artifacts")
                logging.error(f"📁 Artifacts directory contents: {files}")
            else:
                logging.error("📁 Artifacts directory does not exist")
            
            # Check data directory
            if os.path.exists("data"):
                files = os.listdir("data")
                logging.error(f"📁 Data directory contents: {files}")
            else:
                logging.error("📁 Data directory does not exist")
            
            # Check current working directory
            logging.error(f"📁 Current working directory: {os.getcwd()}")
            
            # Check Python path
            logging.error(f"🐍 Python path: {sys.path[:3]}...")  # First 3 entries
            
        except Exception as e:
            logging.error(f"Could not generate debug info: {e}")

    def get_pipeline_status(self):
        """
        Check the current status of pipeline artifacts
        """
        try:
            artifacts_status = {}
            artifacts = self.get_artifact_paths()
            
            for name, path in artifacts.items():
                artifacts_status[name] = {
                    "exists": os.path.exists(path),
                    "path": path,
                    "size": os.path.getsize(path) if os.path.exists(path) else 0,
                    "accessible": True
                }
                
                # Test file accessibility
                if artifacts_status[name]["exists"]:
                    try:
                        with open(path, 'rb') as f:
                            f.read(1)  # Try to read first byte
                    except Exception:
                        artifacts_status[name]["accessible"] = False
            
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
        print("--- Training Complete ---")
        print(f"Best model accuracy: {result['model_accuracy']:.4f}")
        print("Performance of All Models:")
        print(result['model_report'])
        print("="*80)
        print(f"📁 Artifacts created in: ./artifacts/")
        print(f"🔗 MLflow UI: http://localhost:5000")
        print("\n🚀 Ready to run Flask app: python app.py")
        
    except Exception as e:
        print(f"\n❌ Training Pipeline Failed: {str(e)}")
        print("💡 Try running the setup script: setup_and_fix.bat")
        sys.exit(1)