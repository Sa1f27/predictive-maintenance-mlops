# src/pipeline/batch_prediction.py
import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.pipeline.predict_pipeline import PredictPipeline
from datetime import datetime

class BatchPrediction:
    def __init__(self):
        self.predict_pipeline = PredictPipeline()
        
    def predict_batch(self, input_file_path, output_file_path=None):
        """
        Perform batch predictions on a CSV file
        """
        try:
            logging.info(f"Starting batch prediction for: {input_file_path}")
            
            # Read input data
            df = pd.read_csv(input_file_path)
            logging.info(f"Loaded {len(df)} records for batch prediction")
            
            # Validate required columns
            required_columns = [
                "Type", "Air temperature [K]", "Process temperature [K]", 
                "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"
            ]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise CustomException(f"Missing required columns: {missing_columns}", sys)
            
            # Make predictions
            predictions, confidence_scores = self.predict_pipeline.predict(df[required_columns])
            
            # Add predictions to dataframe
            df['Prediction'] = predictions
            df['Failure_Risk'] = ['High' if pred == 1 else 'Low' for pred in predictions]
            if confidence_scores is not None:
                df['Confidence'] = confidence_scores
            df['Prediction_Timestamp'] = datetime.now().isoformat()
            
            # Save results
            if output_file_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file_path = f"predictions/batch_predictions_{timestamp}.csv"
            
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            df.to_csv(output_file_path, index=False)
            
            logging.info(f"Batch predictions saved to: {output_file_path}")
            
            # Summary statistics
            total_predictions = len(predictions)
            high_risk_count = sum(predictions)
            low_risk_count = total_predictions - high_risk_count
            
            summary = {
                "total_predictions": total_predictions,
                "high_risk_count": high_risk_count,
                "low_risk_count": low_risk_count,
                "high_risk_percentage": (high_risk_count / total_predictions) * 100,
                "output_file": output_file_path
            }
            
            logging.info(f"Batch prediction summary: {summary}")
            return summary
            
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    # Example usage
    batch_predictor = BatchPrediction()
    
    # Create sample batch data for testing
    sample_data = pd.DataFrame({
        "Type": ["M", "L", "H", "M", "L"],
        "Air temperature [K]": [298.1, 295.3, 302.5, 299.8, 297.2],
        "Process temperature [K]": [308.6, 305.1, 312.8, 310.2, 306.7],
        "Rotational speed [rpm]": [1551, 1423, 1789, 1634, 1398],
        "Torque [Nm]": [42.8, 38.2, 48.9, 45.1, 36.7],
        "Tool wear [min]": [0, 45, 120, 78, 23]
    })
    
    os.makedirs("sample_data", exist_ok=True)
    sample_data.to_csv("sample_data/sample_batch.csv", index=False)
    
    try:
        result = batch_predictor.predict_batch("sample_data/sample_batch.csv")
        print(f"Batch prediction completed: {result}")
    except Exception as e:
        print(f"Batch prediction failed: {e}")