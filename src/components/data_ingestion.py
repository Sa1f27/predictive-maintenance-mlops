# src/components/data_ingestion.py - NO FALLBACKS VERSION
import sys
import os

# Add the project root directory to the Python path.
# This allows the script to be run directly for testing, resolving the 'src' module not found error.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split

# Imports for standalone execution
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion process")
        try:
            # ORIGINAL PATH FROM YOUR CODE - NO FALLBACKS
            data_path = r'Data\predictive_maintenance.csv'
            
            logging.info(f"Attempting to load dataset from: {data_path}")
            logging.info(f"Current working directory: {os.getcwd()}")
            logging.info(f"Does file exist? {os.path.exists(data_path)}")
            
            # List what's actually in the Data directory
            data_dir = 'Data'
            if os.path.exists(data_dir):
                files_in_data_dir = os.listdir(data_dir)
                logging.info(f"Files in {data_dir} directory: {files_in_data_dir}")
            else:
                logging.error(f"Directory {data_dir} does not exist!")
                raise FileNotFoundError(f"Directory {data_dir} does not exist!")
            
            # Try to load the exact file you specified
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Dataset file not found at: {data_path}")
            
            df = pd.read_csv(data_path)
            logging.info(f"Dataset loaded successfully. Shape: {df.shape}")
            logging.info(f"Columns: {list(df.columns)}")
            logging.info(f"First few rows:\n{df.head()}")
            
            # Show the actual data structure
            logging.info(f"Data types:\n{df.dtypes}")
            logging.info(f"Missing values:\n{df.isnull().sum()}")
            
            # Create artifacts directory
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Raw data saved to: {self.ingestion_config.raw_data_path}")

            # Perform train-test split
            logging.info("Performing train-test split")
            
            # Check if target column exists - NO FALLBACKS
            target_candidates = ['Target', 'Machine failure', 'Failure']
            target_column = None
            
            for candidate in target_candidates:
                if candidate in df.columns:
                    target_column = candidate
                    logging.info(f"Found target column: {target_column}")
                    break
            
            if target_column is None:
                logging.error(f"NO TARGET COLUMN FOUND!")
                logging.error(f"Available columns: {list(df.columns)}")
                logging.error(f"Looking for one of: {target_candidates}")
                raise CustomException(f"No target column found in dataset. Available columns: {list(df.columns)}", sys)
            
            # Check target distribution
            target_distribution = df[target_column].value_counts()
            logging.info(f"Target distribution:\n{target_distribution}")
            
            train_set, test_set = train_test_split(
                df, 
                test_size=0.25, 
                random_state=42, 
                stratify=df[target_column]
            )

            # Save train and test sets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info(f"Train set saved: {self.ingestion_config.train_data_path} (Shape: {train_set.shape})")
            logging.info(f"Test set saved: {self.ingestion_config.test_data_path} (Shape: {test_set.shape})")
            logging.info("Data ingestion completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except FileNotFoundError as e:
            logging.error(f"FILE NOT FOUND ERROR: {str(e)}")
            logging.error(f"Make sure you have the dataset file at: {data_path}")
            raise CustomException(f"Dataset file missing: {str(e)}", sys)
            
        except Exception as e:
            logging.error(f"Data ingestion failed with error: {str(e)}")
            raise CustomException(e, sys)


if __name__ == '__main__':
    logging.info("--- Running Data Ingestion as a standalone script ---")
    logging.info("This will execute the full ingestion, transformation, and training pipeline.")
    
    # 1. Data Ingestion
    obj = DataIngestion()
    train_df_path, test_df_path = obj.initiate_data_ingestion()
    
    # 2. Data Transformation
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_transformation(train_df_path, test_df_path)

    # 3. Model Training
    model_trainer = ModelTrainer()
    accuracy_score, model_report = model_trainer.initiate_model_training(train_arr, test_arr)

    # 4. Print Results
    print("\n" + "="*80)
    print("🎉 STANDALONE RUN COMPLETED! 🎉")
    print("--- Training Complete ---")
    print(f"Best model accuracy: {accuracy_score:.4f}")
    print("Performance of All Models:")
    print(model_report)
    print("="*80)
    print("Note: For a structured pipeline run, please use 'src/pipeline/train_pipeline.py'")