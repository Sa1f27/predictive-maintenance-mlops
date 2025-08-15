# 🔧 Predictive Maintenance System with MLOps Pipeline

> **Industrial IoT Analytics** • **Machine Learning** • **MLOps** • **Production Deployment**

A complete ML system to **predict equipment failures before they happen**, built with **modern MLOps practices**: experiment tracking, model versioning, containerized deployment, and automated workflows.

---

## 🎯 Highlights

* **88–92% accuracy** across multiple equipment types
* **<100ms latency** for real-time predictions
* **End-to-end pipeline**: data ingestion → model training → API deployment
* **Production-ready** FastAPI service with health endpoints and Pydantic validation
* **Fully containerized** via Docker for local & cloud use

---

## 📊 Project Overview

Predictive maintenance allows manufacturers to schedule repairs **before breakdowns**, reducing downtime and costs.
This project processes sensor data (temperature, torque, speed, tool wear) to predict failures using ML models and serves predictions via a web API.

---

## 🛠 Technical Stack

**ML:** scikit-learn, pandas, numpy, SMOTE

**API:** FastAPI, Pydantic

**MLOps:** MLflow (experiment tracking & model registry)

**Containerization:** Docker, Docker Compose

**CI/CD:** GitHub Actions

**Deployment:** AWS ECR + ECS

---

## 📁 Architecture

```
Data Pipeline → Model Training (MLflow) → FastAPI API → Docker → AWS ECS
```

---

## 📸 Screenshots


![Capture](https://github.com/user-attachments/assets/b5ea1ab5-03d2-44e6-8820-187a19caeaef) 

![Capture1](https://github.com/user-attachments/assets/f26830f7-deff-475a-bfeb-f62a10ef6608) 

![Capture3](https://github.com/user-attachments/assets/31c9f394-42b5-43dd-a88c-2dd433af423d) 

![Capture4](https://github.com/user-attachments/assets/d0772238-2de6-4264-ac18-958c9dec3bc4) 

![Capture5](https://github.com/user-attachments/assets/9dbe47c1-21b9-4ef9-8144-6cafc4130858) 

![Capture6](https://github.com/user-attachments/assets/6bc82e51-f6ae-40f0-b5e4-0aeee2da48ce) 

<img width="1878" height="693" alt="Screenshot 2025-08-15 120611" src="https://github.com/user-attachments/assets/afe3a0b4-9599-4a5a-8eb8-fd2dac558420" />

---

## 📈 Model Performance

| Model               | Accuracy  | Precision | Recall | F1-Score |
| ------------------- | --------- | --------- | ------ | -------- |
| Random Forest       | **91.2%** | 89.4%     | 92.1%  | 90.7%    |
| Gradient Boosting   | 89.8%     | 87.3%     | 91.5%  | 89.3%    |
| Logistic Regression | 86.4%     | 84.1%     | 88.7%  | 86.3%    |
| SVM                 | 88.1%     | 85.9%     | 90.2%  | 88.0%    |

**Feature Importance:**

1. Tool Wear (32%)
2. Temperature Differential (24%)
3. Torque Variance (21%)
4. Rotational Speed (15%)
5. Equipment Type (8%)

---

## 🚀 Quick Start

```bash
git clone https://github.com/Sa1f27/predictive-maintenance-mlops.git
cd predictive-maintenance-mlops

# Setup environment
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Train model
python run_pipeline.py --mode train

# Start API
python app.py
```

**Docker Deployment**

```bash
docker-compose up -d --build
```

---

## 🔮 Future Improvements

* Advanced feature engineering (rolling stats, lag features)
* Ensemble/stacking models
* Real-time data streaming with automated retraining
* Advanced monitoring with dashboards

---
