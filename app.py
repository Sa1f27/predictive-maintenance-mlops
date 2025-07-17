import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify
import os
from datetime import datetime

from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import logging

application = Flask(__name__)
app = application

# Initialize prediction pipeline
predict_pipeline = PredictPipeline()

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            data = CustomData(
                Type=request.form.get('Type'),
                Air_temperature=float(request.form.get('Air_temperature_K')),
                Process_temperature=float(request.form.get('Process_temperature_K')),
                Rotational_speed=int(request.form.get('Rotational_speed_rpm')),
                Torque=float(request.form.get('Torque_Nm')),
                Tool_wear=int(request.form.get('Tool_wear_min'))
            )
            
            pred_df = data.get_data_as_data_frame()
            logging.info("Starting prediction")

            results, confidence = predict_pipeline.predict(pred_df)
            logging.info(f"Prediction completed: {results[0]}")
            
            # Enhanced result interpretation
            if results[0] == 1:
                risk_level = "HIGH RISK"
                if confidence and confidence > 0.8:
                    prediction_result = f"⚠️ HIGH RISK: Machine failure predicted with {confidence:.1%} confidence. Immediate maintenance required!"
                else:
                    prediction_result = "⚠️ HIGH RISK: Machine failure predicted. Schedule maintenance immediately."
            else:
                risk_level = "LOW RISK"
                if confidence and confidence > 0.8:
                    prediction_result = f"✅ LOW RISK: Machine operating normally with {confidence:.1%} confidence."
                else:
                    prediction_result = "✅ LOW RISK: Machine operating normally. Continue regular monitoring."
            
            # Log prediction for monitoring (simple logging for now)
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "input_data": pred_df.to_dict('records')[0],
                "prediction": int(results[0]),
                "confidence": float(confidence) if confidence else None,
                "risk_level": risk_level
            }
            logging.info(f"Prediction logged: {log_entry}")
            
            return render_template('home.html', results=prediction_result)
            
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            return render_template('home.html', results=f"Error occurred during prediction: Please check your input values.")

@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        # Check if model files exist
        model_exists = os.path.exists("artifacts/model.pkl")
        preprocessor_exists = os.path.exists("artifacts/preprocessor.pkl")
        
        return jsonify({
            "status": "healthy" if model_exists and preprocessor_exists else "unhealthy",
            "model_loaded": model_exists,
            "preprocessor_loaded": preprocessor_exists,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)