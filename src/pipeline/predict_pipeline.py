import sys
import pandas as pd
import os
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            
            logging.info("Loading model and preprocessor")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            logging.info("Scaling input features")
            data_scaled = preprocessor.transform(features)
            
            logging.info("Making prediction")
            preds = model.predict(data_scaled)
            
            # Get prediction probabilities if available
            try:
                pred_proba = model.predict_proba(data_scaled)
                confidence = pred_proba.max(axis=1)[0]
                logging.info(f"Prediction confidence: {confidence:.4f}")
            except:
                confidence = None
            
            return preds, confidence
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                Type: str,
                Air_temperature: float,
                Process_temperature: float,
                Rotational_speed: int,
                Torque: float,
                Tool_wear: int):

        self.Type = Type
        self.Air_temperature = Air_temperature
        self.Process_temperature = Process_temperature
        self.Rotational_speed = Rotational_speed
        self.Torque = Torque
        self.Tool_wear = Tool_wear

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Type": [self.Type],
                "Air temperature [K]": [self.Air_temperature],
                "Process temperature [K]": [self.Process_temperature],
                "Rotational speed [rpm]": [self.Rotational_speed],
                "Torque [Nm]": [self.Torque],
                "Tool wear [min]": [self.Tool_wear],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)