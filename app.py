from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
import pandas as pd
import os
from datetime import datetime
import traceback
from pydantic import BaseModel

from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import logging
from src.exception import CustomException

# Create FastAPI app
app = FastAPI(
    title="Predictive Maintenance System",
    description="ML-powered equipment failure prediction",
    version="1.0.0"
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

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

# Pydantic models for API
class PredictionRequest(BaseModel):
    Type: str
    Air_temperature: float
    Process_temperature: float
    Rotational_speed: int
    Torque: float
    Tool_wear: int

class PredictionResponse(BaseModel):
    prediction: int
    failure_risk: str
    confidence: float = None
    timestamp: str
    model_version: str = "v1.0.0"

@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup"""
    logging.info("Starting FastAPI application...")
    
    # Check if training is needed
    if not os.path.exists("artifacts/model.pkl"):
        logging.warning("Model artifacts not found. Please run training first:")
        logging.warning("python run_pipeline.py --mode train")
    else:
        initialize_pipeline()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Landing page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/predictdata", response_class=HTMLResponse)
async def predict_form(request: Request):
    """Show prediction form"""
    return templates.TemplateResponse("home.html", {"request": request})

@app.post("/predictdata", response_class=HTMLResponse)
async def predict_datapoint(
    request: Request,
    Type: str = Form(...),
    Air_temperature_K: float = Form(...),
    Process_temperature_K: float = Form(...),
    Rotational_speed_rpm: int = Form(...),
    Torque_Nm: float = Form(...),
    Tool_wear_min: int = Form(...)
):
    """Main prediction endpoint for web form"""
    try:
        # Initialize pipeline if not already done
        if predict_pipeline is None:
            if not initialize_pipeline():
                return templates.TemplateResponse("home.html", {
                    "request": request,
                    "results": "❌ ERROR: Prediction system not available. Please ensure models are trained."
                })

        # Create CustomData object
        try:
            data = CustomData(
                Type=Type,
                Air_temperature=Air_temperature_K,
                Process_temperature=Process_temperature_K,
                Rotational_speed=Rotational_speed_rpm,
                Torque=Torque_Nm,
                Tool_wear=Tool_wear_min
            )
        except (ValueError, TypeError) as e:
            return templates.TemplateResponse("home.html", {
                "request": request,
                "results": "❌ ERROR: Invalid input values. Please check your numeric inputs."
            })
        
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
        
        return templates.TemplateResponse("home.html", {
            "request": request,
            "results": prediction_result
        })
        
    except CustomException as e:
        error_msg = f"❌ Prediction Error: {str(e)}"
        logging.error(error_msg)
        return templates.TemplateResponse("home.html", {
            "request": request,
            "results": error_msg
        })
    
    except Exception as e:
        error_msg = f"❌ Unexpected Error: Please check your input values and try again."
        logging.error(f"Unexpected error in prediction: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        return templates.TemplateResponse("home.html", {
            "request": request,
            "results": error_msg
        })

@app.get("/health")
async def health_check():
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
        
        return health_status
        
    except Exception as e:
        logging.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/api/predict", response_model=PredictionResponse)
async def api_predict(request: PredictionRequest):
    """REST API endpoint for predictions"""
    try:
        if predict_pipeline is None:
            if not initialize_pipeline():
                raise HTTPException(
                    status_code=503, 
                    detail="Prediction system not available. Models not trained or artifacts missing."
                )

        # Create CustomData object
        data = CustomData(
            Type=request.Type,
            Air_temperature=request.Air_temperature,
            Process_temperature=request.Process_temperature,
            Rotational_speed=request.Rotational_speed,
            Torque=request.Torque,
            Tool_wear=request.Tool_wear
        )
        
        # Make prediction
        pred_df = data.get_data_as_data_frame()
        results, confidence = predict_pipeline.predict(pred_df)
        
        # Return response
        return PredictionResponse(
            prediction=int(results[0]),
            failure_risk="High" if results[0] == 1 else "Low",
            confidence=float(confidence) if confidence else None,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logging.error(f"API prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/docs")
async def get_docs():
    """Redirect to API documentation"""
    from fastapi.openapi.docs import get_swagger_ui_html
    return get_swagger_ui_html(openapi_url="/openapi.json", title="API Documentation")

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or default to 8080
    port = int(os.environ.get("PORT", 8080))
    
    print("🚀 Starting Predictive Maintenance FastAPI Application")
    print(f"🌐 Access at: http://localhost:{port}")
    print(f"📋 API Docs at: http://localhost:{port}/docs")
    print(f"🔍 Health Check at: http://localhost:{port}/health")
    
    uvicorn.run(app, host="0.0.0.0", port=port)
