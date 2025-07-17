# src/utils.py - NO FALLBACKS VERSION
import os
import sys
import numpy as np 
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    """Save object to pickle file - NO FALLBACKS"""
    try:
        dir_path = os.path.dirname(file_path)
        
        # Check if directory creation fails
        try:
            os.makedirs(dir_path, exist_ok=True)
        except Exception as e:
            raise CustomException(f"Cannot create directory {dir_path}: {str(e)}", sys)

        # Check if we can actually write to the location
        try:
            with open(file_path, "wb") as file_obj:
                pickle.dump(obj, file_obj)
        except Exception as e:
            raise CustomException(f"Cannot write to file {file_path}: {str(e)}", sys)
            
        # Verify the file was actually created
        if not os.path.exists(file_path):
            raise CustomException(f"File was not created: {file_path}", sys)
            
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            raise CustomException(f"File created but is empty: {file_path}", sys)
            
        logging.info(f"Object saved successfully: {file_path} ({file_size:,} bytes)")

    except Exception as e:
        raise CustomException(f"save_object failed: {str(e)}", sys)


def load_object(file_path):
    """Load object from pickle file - NO FALLBACKS"""
    try:
        # Check file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File does not exist: {file_path}")
        
        # Check file is not empty
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            raise ValueError(f"File exists but is empty: {file_path}")
            
        # Try to load
        try:
            with open(file_path, "rb") as file_obj:
                obj = pickle.load(file_obj)
        except Exception as e:
            raise ValueError(f"Cannot load pickle file {file_path}: {str(e)}")
            
        logging.info(f"Object loaded successfully: {file_path} ({file_size:,} bytes)")
        return obj

    except Exception as e:
        raise CustomException(f"load_object failed: {str(e)}", sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """Evaluate models - NO FALLBACKS, STRICT VALIDATION"""
    try:
        # Validate inputs
        if X_train is None or y_train is None or X_test is None or y_test is None:
            raise ValueError("Input data cannot be None")
            
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError("Training data is empty")
            
        if len(X_test) == 0 or len(y_test) == 0:
            raise ValueError("Test data is empty")
            
        if X_train.shape[1] != X_test.shape[1]:
            raise ValueError(f"Feature mismatch: train {X_train.shape[1]} vs test {X_test.shape[1]}")
        
        if not models:
            raise ValueError("No models provided")
            
        if not param:
            raise ValueError("No parameters provided")

        logging.info(f"Evaluating {len(models)} models")
        logging.info(f"Training data shape: {X_train.shape}")
        logging.info(f"Test data shape: {X_test.shape}")
        logging.info(f"Target distribution - train: {np.bincount(y_train.astype(int))}")
        logging.info(f"Target distribution - test: {np.bincount(y_test.astype(int))}")

        report = {}

        for model_name, model in models.items():
            logging.info(f"Training model: {model_name}")
            
            # Check if parameters exist for this model
            if model_name not in param:
                raise KeyError(f"No parameters defined for model: {model_name}")
            
            model_params = param[model_name]
            
            try:
                # Grid search
                gs = GridSearchCV(
                    estimator=model,
                    param_grid=model_params,
                    cv=3,
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=0
                )
                
                gs.fit(X_train, y_train)
                
                # Check if grid search found valid results
                if gs.best_score_ is None:
                    raise ValueError(f"Grid search failed for {model_name}")

                # Train final model
                model.set_params(**gs.best_params_)
                model.fit(X_train, y_train)

                # Make predictions
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                # Calculate scores
                train_accuracy = accuracy_score(y_train, y_train_pred)
                test_accuracy = accuracy_score(y_test, y_test_pred)

                # Validate scores
                if train_accuracy < 0 or train_accuracy > 1:
                    raise ValueError(f"Invalid train accuracy for {model_name}: {train_accuracy}")
                    
                if test_accuracy < 0 or test_accuracy > 1:
                    raise ValueError(f"Invalid test accuracy for {model_name}: {test_accuracy}")

                report[model_name] = test_accuracy
                
                logging.info(f"{model_name} completed - Train: {train_accuracy:.4f}, Test: {test_accuracy:.4f}")
                logging.info(f"{model_name} best params: {gs.best_params_}")
                
            except Exception as e:
                logging.error(f"Model {model_name} failed: {str(e)}")
                raise CustomException(f"Model evaluation failed for {model_name}: {str(e)}", sys)

        # Validate final report
        if not report:
            raise ValueError("No models were successfully evaluated")
            
        if len(report) != len(models):
            failed_models = set(models.keys()) - set(report.keys())
            logging.warning(f"Some models failed: {failed_models}")

        logging.info(f"Model evaluation completed. Results: {report}")
        return report

    except Exception as e:
        logging.error(f"evaluate_models failed: {str(e)}")
        raise CustomException(f"Model evaluation failed: {str(e)}", sys)


def validate_data_quality(df, required_columns):
    """Strict data quality validation - NO FALLBACKS"""
    try:
        if df is None:
            raise ValueError("DataFrame is None")
            
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        # Check required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for all-null columns
        null_cols = df[required_columns].isnull().all()
        completely_null = null_cols[null_cols].index.tolist()
        if completely_null:
            raise ValueError(f"Columns are completely null: {completely_null}")
        
        # Check data types
        for col in required_columns:
            if col == 'Type':
                if df[col].dtype != 'object':
                    raise ValueError(f"Column {col} should be string/object type, got {df[col].dtype}")
            else:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    raise ValueError(f"Column {col} should be numeric, got {df[col].dtype}")
        
        # Check for reasonable data ranges (no fallbacks, just validation)
        if 'Air temperature [K]' in df.columns:
            temp_range = df['Air temperature [K]'].agg(['min', 'max'])
            if temp_range['min'] < 200 or temp_range['max'] > 400:
                logging.warning(f"Unusual air temperature range: {temp_range['min']:.1f} - {temp_range['max']:.1f} K")
        
        logging.info(f"Data quality validation passed for {len(df)} rows, {len(required_columns)} columns")
        return True
        
    except Exception as e:
        raise CustomException(f"Data quality validation failed: {str(e)}", sys)


# REMOVED: create_sample_data function - NO FALLBACKS ALLOWED

if __name__ == "__main__":
    print("❌ NO FALLBACK UTILS - This will only work with real data")
    print("📊 Make sure you have 'Data/predictive_maintenance.csv' in your project")