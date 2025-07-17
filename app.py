# app.py - FIXED VERSION
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify
import os
from datetime import datetime
import traceback

from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import logging
from src.exception import CustomException

application = Flask(__name__)
app = application

# Initialize prediction pipeline globally
predict_pipeline = None

def initialize_pipeline():
    """Initialize prediction pipeline with error handling"""
    global predict_pipeline
    try:
        predict_pipeline = PredictPipeline()
        logging.info("Prediction pipeline initialized successfully")
        return True
    except Exception as e:
        logging.error(f"Failed to initialize prediction pipeline: {str(e)}")
        return False

@app.route('/')
def index():
    """Landing page"""
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    """Main prediction endpoint"""
    if request.method == 'GET':
        return render_template('home.html')
    
    try:
        # Initialize pipeline if not already done
        if predict_pipeline is None:
            if not initialize_pipeline():
                return render_template('home.html', 
                    results="❌ ERROR: Prediction system not available. Please ensure models are trained.")

        # Extract and validate form data
        form_data = {
            'Type': request.form.get('Type'),
            'Air_temperature': request.form.get('Air_temperature_K'),
            'Process_temperature': request.form.get('Process_temperature_K'),
            'Rotational_speed': request.form.get('Rotational_speed_rpm'),
            'Torque': request.form.get('Torque_Nm'),
            'Tool_wear': request.form.get('Tool_wear_min')
        }
        
        # Validate required fields
        missing_fields = [key for key, value in form_data.items() if not value]
        if missing_fields:
            return render_template('home.html', 
                results=f"❌ ERROR: Missing required fields: {', '.join(missing_fields)}")

        # Convert and validate numeric fields
        try:
            data = CustomData(
                Type=form_data['Type'],
                Air_temperature=float(form_data['Air_temperature']),
                Process_temperature=float(form_data['Process_temperature']),
                Rotational_speed=int(form_data['Rotational_speed']),
                Torque=float(form_data['Torque']),
                Tool_wear=int(form_data['Tool_wear'])
            )
        except (ValueError, TypeError) as e:
            return render_template('home.html', 
                results="❌ ERROR: Invalid input values. Please check your numeric inputs.")
        
        # Convert to DataFrame
        pred_df = data.get_data_as_data_frame()
        logging.info(f"Prediction input: {pred_df.to_dict('records')[0]}")

        # Make prediction
        logging.info("Starting prediction")
        results, confidence = predict_pipeline.predict(pred_df)
        
        # Log successful prediction
        logging.info(f"Prediction completed: {results[0]}, Confidence: {confidence}")
        
        # Format result message
        if results[0] == 1:
            risk_level = "HIGH RISK"
            if confidence and confidence > 0.8:
                prediction_result = f"⚠️ HIGH RISK: Machine failure predicted with {confidence:.1%} confidence. Immediate maintenance recommended!"
            else:
                prediction_result = "⚠️ HIGH RISK: Machine failure predicted. Schedule maintenance immediately."
        else:
            risk_level = "LOW RISK"
            if confidence and confidence > 0.8:
                prediction_result = f"✅ LOW RISK: Machine operating normally with {confidence:.1%} confidence."
            else:
                prediction_result = "✅ LOW RISK: Machine operating normally. Continue regular monitoring."
        
        # Log prediction for monitoring
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "input_data": pred_df.to_dict('records')[0],
            "prediction": int(results[0]),
            "confidence": float(confidence) if confidence else None,
            "risk_level": risk_level
        }
        logging.info(f"Prediction logged: {log_entry}")
        
        return render_template('home.html', results=prediction_result)
        
    except CustomException as e:
        error_msg = f"❌ Prediction Error: {str(e)}"
        logging.error(error_msg)
        return render_template('home.html', results=error_msg)
    
    except Exception as e:
        error_msg = f"❌ Unexpected Error: Please check your input values and try again."
        logging.error(f"Unexpected error in prediction: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        return render_template('home.html', results=error_msg)

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    try:
        health_status = {
            "status": "unknown",
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }
        
        # Check if artifacts exist
        artifacts = {
            "model": "artifacts/model.pkl",
            "preprocessor": "artifacts/preprocessor.pkl",
            "train_data": "artifacts/train.csv",
            "test_data": "artifacts/test.csv",
            "raw_data": "artifacts/data.csv"
        }
        
        all_healthy = True
        for name, path in artifacts.items():
            exists = os.path.exists(path)
            size = os.path.getsize(path) if exists else 0
            health_status["components"][name] = {
                "exists": exists,
                "size_bytes": size,
                "path": path
            }
            if not exists:
                all_healthy = False
        
        # Check pipeline initialization
        pipeline_status = predict_pipeline is not None
        health_status["components"]["prediction_pipeline"] = {
            "initialized": pipeline_status,
            "ready": pipeline_status
        }
        
        if not pipeline_status:
            all_healthy = False
        
        health_status["status"] = "healthy" if all_healthy else "degraded"
        
        return jsonify(health_status), 200 if all_healthy else 503
        
    except Exception as e:
        logging.error(f"Health check failed: {str(e)}")
        return jsonify({
            "status": "error", 
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """REST API endpoint for predictions"""
    try:
        if predict_pipeline is None:
            if not initialize_pipeline():
                return jsonify({
                    "error": "Prediction system not available",
                    "message": "Models not trained or artifacts missing"
                }), 503

        # Get JSON data
        json_data = request.get_json()
        if not json_data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Create CustomData object
        data = CustomData(
            Type=json_data.get('Type'),
            Air_temperature=float(json_data.get('Air_temperature')),
            Process_temperature=float(json_data.get('Process_temperature')),
            Rotational_speed=int(json_data.get('Rotational_speed')),
            Torque=float(json_data.get('Torque')),
            Tool_wear=int(json_data.get('Tool_wear'))
        )
        
        # Make prediction
        pred_df = data.get_data_as_data_frame()
        results, confidence = predict_pipeline.predict(pred_df)
        
        # Return JSON response
        response = {
            "prediction": int(results[0]),
            "failure_risk": "High" if results[0] == 1 else "Low",
            "confidence": float(confidence) if confidence else None,
            "timestamp": datetime.now().isoformat(),
            "model_version": "v1.0.0"
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logging.error(f"API prediction failed: {str(e)}")
        return jsonify({
            "error": "Prediction failed",
            "message": str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    logging.error(f"Internal server error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    # Initialize pipeline on startup
    logging.info("Starting Flask application...")
    
    # Check if training is needed
    if not os.path.exists("artifacts/model.pkl"):
        logging.warning("Model artifacts not found. Please run training first:")
        logging.warning("python run_pipeline.py --mode train")
    else:
        initialize_pipeline()
    
    # Run the application
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)