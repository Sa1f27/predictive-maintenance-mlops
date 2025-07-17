# src/components/data_transformation.py - FIXED VERSION
import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    # FIXED: Changed 'artificats' to 'artifacts'
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Create the data transformation pipeline
        """
        try:
            # Define numerical and categorical features
            numerical_columns = [
                "Air temperature [K]",
                "Process temperature [K]", 
                "Rotational speed [rpm]",
                "Torque [Nm]",
                "Tool wear [min]"
            ]
            
            categorical_columns = ["Type"]

            # Create numerical pipeline
            numerical_pipeline = Pipeline([
                ("scaler", StandardScaler())
            ])
            
            # Create categorical pipeline
            categorical_pipeline = Pipeline([
                ("one_hot_encoder", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))
            ])
            
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Combine pipelines
            preprocessor = ColumnTransformer([
                ("num_pipeline", numerical_pipeline, numerical_columns),
                ("cat_pipeline", categorical_pipeline, categorical_columns)
            ])
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_transformation(self, train_path, test_path):
        """
        Perform data transformation on train and test datasets
        """
        try:
            # Read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test data completed")
            logging.info(f"Train data shape: {train_df.shape}")
            logging.info(f"Test data shape: {test_df.shape}")
            
            # Display column information for debugging
            logging.info(f"Train columns: {list(train_df.columns)}")

            # Get preprocessing object
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            # Define target column and columns to drop
            target_column_candidates = ['Target', 'Machine failure', 'Failure']
            target_column = None
            
            for candidate in target_column_candidates:
                if candidate in train_df.columns:
                    target_column = candidate
                    break
            
            if target_column is None:
                raise CustomException("No target column found in the dataset", sys)
            
            logging.info(f"Using target column: {target_column}")
            
            # Columns to drop (non-predictive features)
            drop_columns = ['UDI', 'Product ID', 'Failure Type']
            # Only drop columns that actually exist
            drop_columns = [col for col in drop_columns if col in train_df.columns]
            
            logging.info(f"Dropping columns: {drop_columns}")

            # Separate features and target variable
            input_feature_train_df = train_df.drop(columns=[target_column] + drop_columns, axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column] + drop_columns, axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info(f"Input features shape - Train: {input_feature_train_df.shape}, Test: {input_feature_test_df.shape}")
            logging.info(f"Target distribution - Train: {target_feature_train_df.value_counts().to_dict()}")

            # Apply preprocessing object on training and testing datasets
            logging.info("Applying preprocessing object on training dataframe")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            
            logging.info("Applying preprocessing object on testing dataframe")
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine features and target
            train_arr = np.c_[
                input_feature_train_arr, 
                np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, 
                np.array(target_feature_test_df)
            ]

            logging.info(f"Final arrays - Train: {train_arr.shape}, Test: {test_arr.shape}")

            # Save preprocessing object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            logging.info(f"Saved preprocessing object to: {self.data_transformation_config.preprocessor_obj_file_path}")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            logging.error(f"Data transformation failed: {str(e)}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Test the data transformation
    from src.components.data_ingestion import DataIngestion
    
    # First run data ingestion
    data_ingestion = DataIngestion()
    train_path, test_path = data_ingestion.initiate_data_ingestion()
    
    # Then run data transformation
    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = data_transformation.initiate_transformation(train_path, test_path)
    
    print(f"Data transformation completed successfully!")
    print(f"Train array shape: {train_arr.shape}")
    print(f"Test array shape: {test_arr.shape}")
    print(f"Preprocessor saved at: {preprocessor_path}")