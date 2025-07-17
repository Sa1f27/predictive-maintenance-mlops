# run_pipeline.py - Master Pipeline Runner
import os
import sys
import argparse
import time
from datetime import datetime

def run_training_pipeline():
    """Run the complete training pipeline"""
    print("🚀 Starting Training Pipeline...")
    start_time = time.time()
    
    try:
        from src.pipeline.train_pipeline import TrainingPipeline
        pipeline = TrainingPipeline()
        result = pipeline.start_training()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ Training Pipeline Completed Successfully!")
        print(f"⏱️  Duration: {duration:.2f} seconds")
        print(f"🎯 Best Model Accuracy: {result['model_accuracy']:.4f}")
        return True
        
    except Exception as e:
        print(f"❌ Training Pipeline Failed: {e}")
        return False

def run_model_evaluation():
    """Run model evaluation"""
    print("📊 Starting Model Evaluation...")
    
    try:
        from src.pipeline.model_evaluation import ModelEvaluator
        evaluator = ModelEvaluator()
        report = evaluator.evaluate_model()
        
        print(f"✅ Model Evaluation Completed!")
        print(f"🎯 ROC AUC Score: {report['roc_auc_score']:.4f}")
        return True
        
    except Exception as e:
        print(f"❌ Model Evaluation Failed: {e}")
        return False

def run_batch_prediction(input_file=None):
    """Run batch prediction"""
    print("🔮 Starting Batch Prediction...")
    
    try:
        from src.pipeline.batch_prediction import BatchPrediction
        batch_predictor = BatchPrediction()
        
        if input_file is None:
            input_file = "sample_data/sample_batch.csv"
            
        result = batch_predictor.predict_batch(input_file)
        
        print(f"✅ Batch Prediction Completed!")
        print(f"📊 Processed {result['total_predictions']} records")
        print(f"⚠️  High Risk: {result['high_risk_count']} ({result['high_risk_percentage']:.1f}%)")
        return True
        
    except Exception as e:
        print(f"❌ Batch Prediction Failed: {e}")
        return False

def start_mlflow_server():
    """Start MLflow tracking server"""
    print("🔬 Starting MLflow Server...")
    try:
        os.system("mlflow server --host 0.0.0.0 --port 5000 &")
        print("✅ MLflow Server started at http://localhost:5000")
        return True
    except Exception as e:
        print(f"❌ MLflow Server failed: {e}")
        return False

def start_flask_app():
    """Start Flask application"""
    print("🌐 Starting Flask Application...")
    try:
        os.system("python app.py")
        return True
    except Exception as e:
        print(f"❌ Flask App failed: {e}")
        return False

def check_artifacts():
    """Check if all required artifacts exist"""
    required_artifacts = [
        "artifacts/model.pkl",
        "artifacts/preprocessor.pkl",
        "artifacts/train.csv",
        "artifacts/test.csv"
    ]
    
    missing = []
    for artifact in required_artifacts:
        if not os.path.exists(artifact):
            missing.append(artifact)
    
    if missing:
        print(f"⚠️  Missing artifacts: {missing}")
        return False
    else:
        print("✅ All artifacts present")
        return True

def main():
    parser = argparse.ArgumentParser(description="MLOps Pipeline Runner")
    parser.add_argument("--mode", choices=["train", "evaluate", "predict", "serve", "mlflow", "full"], 
                       default="full", help="Pipeline mode to run")
    parser.add_argument("--input-file", help="Input file for batch prediction")
    parser.add_argument("--skip-training", action="store_true", help="Skip training if artifacts exist")
    
    args = parser.parse_args()
    
    print("="*80)
    print("🔧 PREDICTIVE MAINTENANCE MLOPS PIPELINE")
    print("="*80)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Mode: {args.mode.upper()}")
    print("="*80)
    
    if args.mode == "train" or args.mode == "full":
        if args.skip_training and check_artifacts():
            print("⏭️  Skipping training - artifacts already exist")
        else:
            success = run_training_pipeline()
            if not success:
                print("❌ Pipeline failed at training stage")
                sys.exit(1)
    
    if args.mode == "evaluate" or args.mode == "full":
        if check_artifacts():
            run_model_evaluation()
        else:
            print("⚠️  Skipping evaluation - missing artifacts")
    
    if args.mode == "predict":
        if check_artifacts():
            run_batch_prediction(args.input_file)
        else:
            print("⚠️  Cannot run prediction - missing artifacts")
    
    if args.mode == "mlflow":
        start_mlflow_server()
    
    if args.mode == "serve":
        if check_artifacts():
            print("🌐 Starting web application...")
            start_flask_app()
        else:
            print("⚠️  Cannot start app - missing artifacts")
    
    if args.mode == "full":
        print("\n🎉 Full pipeline completed!")
        print("🔗 MLflow UI: http://localhost:5000")
        print("🌐 Flask App: python app.py")
        print("📊 Evaluation plots: ./evaluation_plots/")
    
    print("="*80)
    print(f"✅ Pipeline finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

if __name__ == "__main__":
    main()