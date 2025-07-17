
# src/mlops/mlflow_manager.py
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import os
import sys
from src.exception import CustomException
from src.logger import logging

class MLflowManager:
    def __init__(self, experiment_name="predictive_maintenance", tracking_uri="http://localhost:5000"):
        """
        Initialize MLflow tracking
        """
        try:
            self.experiment_name = experiment_name
            self.tracking_uri = tracking_uri
            
            # Set tracking URI
            mlflow.set_tracking_uri(self.tracking_uri)
            
            # Create or set experiment
            try:
                experiment = mlflow.get_experiment_by_name(self.experiment_name)
                if experiment is None:
                    mlflow.create_experiment(self.experiment_name)
                    logging.info(f"Created new MLflow experiment: {self.experiment_name}")
                mlflow.set_experiment(self.experiment_name)
                logging.info(f"Using MLflow experiment: {self.experiment_name}")
            except Exception as e:
                logging.warning(f"MLflow experiment setup failed: {e}")
                
        except Exception as e:
            logging.warning(f"MLflow initialization failed: {e}")
            self.mlflow_available = False
    
    def log_model_training(self, model, model_name, metrics, params, artifacts_dict=None):
        """
        Log model training to MLflow
        """
        try:
            with mlflow.start_run():
                # Log parameters
                mlflow.log_params(params)
                
                # Log metrics
                mlflow.log_metrics(metrics)
                
                # Log model
                mlflow.sklearn.log_model(
                    sk_model=model,
                    name="model",
                    registered_model_name=f"maintenance_predictor_{model_name.lower().replace(' ', '_')}"
                )
                
                # Log additional artifacts if provided
                if artifacts_dict:
                    for artifact_name, artifact_path in artifacts_dict.items():
                        if os.path.exists(artifact_path):
                            mlflow.log_artifact(artifact_path, name=artifact_name)
                
                run_id = mlflow.active_run().info.run_id
                logging.info(f"Model {model_name} logged to MLflow with run_id: {run_id}")
                return run_id
                
        except Exception as e:
            logging.warning(f"MLflow logging failed: {e}")
            return None
    
    def register_best_model(self, model_name, run_id, stage="Staging"):
        """
        Register the best model to MLflow Model Registry
        """
        try:
            client = MlflowClient()
            model_uri = f"runs:/{run_id}/model"
            
            # Register model
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=f"best_maintenance_predictor"
            )
            
            # Transition to staging
            client.transition_model_version_stage(
                name="best_maintenance_predictor",
                version=registered_model.version,
                stage=stage
            )
            
            logging.info(f"Model registered and moved to {stage} stage")
            return registered_model.version
            
        except Exception as e:
            logging.warning(f"Model registration failed: {e}")
            return None