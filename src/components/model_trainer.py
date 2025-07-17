# src/components/model_trainer.py (WITH MLFLOW)
import os
import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models
from src.mlops.mlflow_manager import MLflowManager

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        # Initialize MLflow manager
        try:
            self.mlflow_manager = MLflowManager()
            self.mlflow_enabled = True
            logging.info("MLflow integration enabled")
        except Exception as e:
            logging.warning(f"MLflow initialization failed: {e}. Continuing without MLflow.")
            self.mlflow_enabled = False

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input and target feature")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestClassifier(random_state=42),
                "SVM": SVC(random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "Logistic Regression": LogisticRegression(random_state=42),
                "KNN": KNeighborsClassifier()
            }

            params = {
                "Random Forest": {
                    'n_estimators': [50, 100],
                    'max_depth': [10, 20, None]
                },
                "SVM": {
                    'C': [1, 10],
                    'kernel': ['linear', 'rbf']
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01],
                    'n_estimators': [50, 100]
                },
                "Logistic Regression": {
                    'C': [0.1, 1, 10],
                    'solver': ['liblinear']
                },
                "KNN": {
                    'n_neighbors': [3, 5],
                    'weights': ['uniform', 'distance']
                }
            }

            model_report: dict = evaluate_models(
                X_train=X_train, 
                y_train=y_train, 
                X_test=X_test, 
                y_test=y_test,
                models=models, 
                param=params
            )
            
            # Get best model score and name
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            # Add this debug logging
            logging.info(f"Model Report: {model_report}")
            logging.info(f"Best Model Score: {best_model_score}")
            logging.info(f"Best Model Name: {best_model_name}")
            
            if best_model_score < 0.1:
                raise CustomException("No best model found with acceptable performance", sys)

            logging.info(f"Best found model: {best_model_name} with score: {best_model_score}")

            # Re-train best model with best parameters to get proper predictions
            best_params = params[best_model_name]
            from sklearn.model_selection import GridSearchCV
            
            gs = GridSearchCV(models[best_model_name], best_params, cv=3, scoring='accuracy')
            gs.fit(X_train, y_train)
            best_model = gs.best_estimator_

            # Make predictions for detailed metrics
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # Calculate comprehensive metrics
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            precision = precision_score(y_test, y_test_pred, average='weighted')
            recall = recall_score(y_test, y_test_pred, average='weighted')
            f1 = f1_score(y_test, y_test_pred, average='weighted')

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Log metrics
            logging.info(f"Best Model: {best_model_name}")
            logging.info(f"Best Parameters: {gs.best_params_}")
            logging.info(f"Train Accuracy: {train_accuracy:.4f}")
            logging.info(f"Test Accuracy: {test_accuracy:.4f}")
            logging.info(f"Precision: {precision:.4f}")
            logging.info(f"Recall: {recall:.4f}")
            logging.info(f"F1-Score: {f1:.4f}")

            # MLflow logging
            if self.mlflow_enabled:
                try:
                    # Prepare metrics for MLflow
                    metrics = {
                        'train_accuracy': train_accuracy,
                        'test_accuracy': test_accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'model_score': best_model_score
                    }
                    
                    # Prepare parameters for MLflow
                    mlflow_params = {
                        'model_type': best_model_name,
                        'train_size': len(X_train),
                        'test_size': len(X_test),
                        'features_count': X_train.shape[1],
                        'random_state': 42
                    }
                    
                    # Add best hyperparameters
                    for param, value in gs.best_params_.items():
                        mlflow_params[f'best_{param}'] = value
                    
                    # Log all models to MLflow for comparison
                    for model_name, model_score in model_report.items():
                        temp_metrics = metrics.copy()
                        temp_metrics['test_accuracy'] = model_score
                        temp_params = mlflow_params.copy()
                        temp_params['model_type'] = model_name
                        
                        run_id = self.mlflow_manager.log_model_training(
                            model=models[model_name] if model_name != best_model_name else best_model,
                            model_name=model_name,
                            metrics=temp_metrics,
                            params=temp_params,
                            artifacts_dict={
                                "model_artifacts": "artifacts"
                            }
                        )
                        
                        # Register the best model
                        if model_name == best_model_name and run_id:
                            self.mlflow_manager.register_best_model(
                                model_name=best_model_name,
                                run_id=run_id,
                                stage="Production"
                            )
                    
                    logging.info("All models logged to MLflow successfully")
                    
                except Exception as e:
                    logging.warning(f"MLflow logging failed: {e}")

            return test_accuracy
            
        except Exception as e:
            raise CustomException(e, sys)