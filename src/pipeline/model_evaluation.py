# src/pipeline/model_evaluation.py
import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class ModelEvaluator:
    def __init__(self):
        self.model_path = "artifacts/model.pkl"
        self.preprocessor_path = "artifacts/preprocessor.pkl"
        self.test_data_path = "artifacts/test.csv"
        
    def load_artifacts(self):
        """Load model, preprocessor, and test data"""
        try:
            self.model = load_object(self.model_path)
            self.preprocessor = load_object(self.preprocessor_path)
            self.test_data = pd.read_csv(self.test_data_path)
            logging.info("All artifacts loaded successfully")
        except Exception as e:
            raise CustomException(e, sys)
    
    def evaluate_model(self):
        """Comprehensive model evaluation"""
        try:
            self.load_artifacts()
            
            # Prepare test data
            target_column = "Target"
            X_test = self.test_data.drop(columns=[target_column])
            y_test = self.test_data[target_column]
            
            # Transform test data
            X_test_scaled = self.preprocessor.transform(X_test)
            
            # Make predictions
            y_pred = self.model.predict(X_test_scaled)
            y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
            
            # Generate evaluation report
            evaluation_report = {
                "classification_report": classification_report(y_test, y_pred, output_dict=True),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
                "roc_auc_score": roc_auc_score(y_test, y_pred_proba),
                "test_samples": len(y_test),
                "feature_names": X_test.columns.tolist()
            }
            
            # Save evaluation plots
            self.create_evaluation_plots(y_test, y_pred, y_pred_proba)
            
            logging.info("Model evaluation completed successfully")
            return evaluation_report
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def create_evaluation_plots(self, y_test, y_pred, y_pred_proba):
        """Create and save evaluation plots"""
        try:
            os.makedirs("evaluation_plots", exist_ok=True)
            
            # Confusion Matrix
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig("evaluation_plots/confusion_matrix.png")
            plt.close()
            
            # ROC Curve
            plt.figure(figsize=(8, 6))
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.savefig("evaluation_plots/roc_curve.png")
            plt.close()
            
            # Prediction Distribution
            plt.figure(figsize=(10, 6))
            plt.subplot(1, 2, 1)
            plt.hist(y_pred_proba[y_test == 0], alpha=0.7, label='No Failure', bins=20)
            plt.hist(y_pred_proba[y_test == 1], alpha=0.7, label='Failure', bins=20)
            plt.xlabel('Prediction Probability')
            plt.ylabel('Frequency')
            plt.title('Prediction Probability Distribution')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            class_counts = pd.Series(y_pred).value_counts()
            plt.pie(class_counts.values, labels=['No Failure', 'Failure'], autopct='%1.1f%%')
            plt.title('Prediction Distribution')
            
            plt.tight_layout()
            plt.savefig("evaluation_plots/prediction_analysis.png")
            plt.close()
            
            logging.info("Evaluation plots saved to evaluation_plots/")
            
        except Exception as e:
            logging.warning(f"Failed to create evaluation plots: {e}")

if __name__ == "__main__":
    try:
        evaluator = ModelEvaluator()
        report = evaluator.evaluate_model()
        
        print("\n" + "="*60)
        print("📊 MODEL EVALUATION REPORT")
        print("="*60)
        print(f"🎯 ROC AUC Score: {report['roc_auc_score']:.4f}")
        print(f"📈 Test Samples: {report['test_samples']}")
        print(f"📊 Accuracy: {report['classification_report']['accuracy']:.4f}")
        print(f"📊 Precision: {report['classification_report']['weighted avg']['precision']:.4f}")
        print(f"📊 Recall: {report['classification_report']['weighted avg']['recall']:.4f}")
        print(f"📊 F1-Score: {report['classification_report']['weighted avg']['f1-score']:.4f}")
        print("="*60)
        print("📁 Evaluation plots saved in: ./evaluation_plots/")
        
    except Exception as e:
        print(f"❌ Model evaluation failed: {e}")