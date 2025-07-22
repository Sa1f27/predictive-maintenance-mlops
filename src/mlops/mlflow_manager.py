# src/mlops/mlflow_manager.py
import mlflow
import mlflow.sklearn
from src.logger import logging
from src.exception import CustomException
import sys

class MLflowManager:
    def __init__(self, tracking_uri="http://localhost:5000", experiment_name="predictive_maintenance"):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self._initialize_mlflow()

    def _initialize_mlflow(self):
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            if not mlflow.get_experiment_by_name(self.experiment_name):
                mlflow.create_experiment(self.experiment_name)
            mlflow.set_experiment(self.experiment_name)
            logging.info(f"MLflow initialized. Tracking URI: {self.tracking_uri}, Experiment: {self.experiment_name}")
        except Exception as e:
            logging.error(f"Failed to initialize MLflow: {e}")
            raise CustomException(e, sys)

    def log_model_training(self, model, model_name, metrics, params, artifacts_dict=None, input_example=None, signature=None):
        try:
            with mlflow.start_run(run_name=f"{model_name}_training") as run:
                logging.info(f"Starting MLflow run for model: {model_name}")
                if params:
                    mlflow.log_params(params)
                if metrics:
                    mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=f"maintenance_{model_name.lower().replace(' ', '_')}"
                )
                if artifacts_dict:
                    for key, path in artifacts_dict.items():
                        mlflow.log_artifacts(path, artifact_path=key)
                
                run_id = run.info.run_id
                logging.info(f"MLflow run for {model_name} completed. Run ID: {run_id}")
                return run_id
        except Exception as e:
            logging.error(f"MLflow logging failed for model {model_name}: {e}")
            raise CustomException(e, sys)

    def register_best_model(self, model_name, run_id, stage="Production"):
        try:
            registered_model_name = f"maintenance_{model_name.lower().replace(' ', '_')}"
            client = mlflow.tracking.MlflowClient()
            latest_versions = client.get_latest_versions(name=registered_model_name, stages=["None", "Staging", "Archived"])
            
            if latest_versions:
                client.transition_model_version_stage(
                    name=registered_model_name,
                    version=latest_versions[0].version,
                    stage=stage
                )
                logging.info(f"Registered model '{registered_model_name}' version {latest_versions[0].version} to stage '{stage}'")
        except Exception as e:
            logging.error(f"Failed to register model {model_name}: {e}")
            raise CustomException(e, sys)