# src/components/model_trainer.py (UPDATED WITH FULL MLFLOW INTEGRATION)
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
from src.utils import save_object, evaluate_models, load_object
from src.mlops.mlflow_manager import MLflowManager
from mlflow.models.signature import infer_signature
from imblearn.over_sampling import SMOTE

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
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

            # Apply SMOTE to the training data to handle class imbalance
            logging.info("Applying SMOTE to balance the training data")
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            logging.info(f"Shape of training data after SMOTE: X_train={X_train.shape}")

            models = {
                "Random Forest": RandomForestClassifier(random_state=42),
                "SVM": SVC(probability=True, random_state=42),
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

            logging.info(f"Full model evaluation report: {model_report}")

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]
            logging.info(f"Best Model: {best_model_name} with score: {best_model_score}")

            if best_model_score < 0.1:
                raise CustomException("No acceptable model found", sys)

            from sklearn.model_selection import GridSearchCV
            gs = GridSearchCV(models[best_model_name], params[best_model_name], cv=3, scoring='accuracy')
            gs.fit(X_train, y_train)
            best_model = gs.best_estimator_

            # --- ADDED: Feature Importance Calculation ---
            if hasattr(best_model, 'feature_importances_'):
                try:
                    preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
                    preprocessing_obj = load_object(file_path=preprocessor_path)
                    
                    # Get feature names from the preprocessor
                    num_feature_names = preprocessing_obj.named_transformers_['num_pipeline'].get_feature_names_out().tolist()
                    cat_feature_names = preprocessing_obj.named_transformers_['cat_pipeline']['one_hot_encoder'].get_feature_names_out().tolist()
                    all_feature_names = num_feature_names + cat_feature_names

                    importances = best_model.feature_importances_
                    feature_importance_df = pd.DataFrame(
                        list(zip(all_feature_names, importances)),
                        columns=['feature', 'importance']
                    ).sort_values(by='importance', ascending=False)

                    logging.info("--- Feature Importances ---")
                    logging.info(f"\n{feature_importance_df}")

                except Exception as e:
                    logging.warning(f"Could not calculate and log feature importances: {e}")

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            precision = precision_score(y_test, y_test_pred, average='weighted')
            recall = recall_score(y_test, y_test_pred, average='weighted')
            f1 = f1_score(y_test, y_test_pred, average='weighted')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info(f"Best Parameters: {gs.best_params_}")
            logging.info(f"Test Accuracy: {test_accuracy:.4f}")
            logging.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

            if self.mlflow_enabled:
                try:
                    for model_name, model_score in model_report.items():
                        model_obj = models[model_name] if model_name != best_model_name else best_model

                        # Input signature
                        input_example = pd.DataFrame(X_test[:1], columns=[f'feature_{i}' for i in range(X_test.shape[1])])
                        signature = infer_signature(X_test, model_obj.predict(X_test))

                        # Metrics
                        metrics = {
                            'train_accuracy': train_accuracy,
                            'test_accuracy': model_score,
                            'precision': precision,
                            'recall': recall,
                            'f1_score': f1
                        }

                        # Params
                        mlflow_params = {
                            'model_type': model_name,
                            'train_size': len(X_train),
                            'test_size': len(X_test),
                            'features_count': X_train.shape[1],
                            'random_state': 42
                        }

                        if model_name == best_model_name:
                            for param, value in gs.best_params_.items():
                                mlflow_params[f'best_{param}'] = value

                        run_id = self.mlflow_manager.log_model_training(
                            model=model_obj,
                            model_name=model_name,
                            metrics=metrics,
                            params=mlflow_params,
                            artifacts_dict={
                                "evaluation_plots": "evaluation_plots",
                                "model_artifacts": "artifacts"
                            },
                            input_example=input_example,
                            signature=signature
                        )

                        if model_name == best_model_name and run_id:
                            self.mlflow_manager.register_best_model(
                                model_name=best_model_name,
                                run_id=run_id,
                                stage="Production"
                            )

                except Exception as e:
                    logging.warning(f"MLflow logging failed: {e}")

            return test_accuracy, model_report

        except Exception as e:
            raise CustomException(e, sys)
        