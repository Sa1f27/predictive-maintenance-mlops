

# 🔧 Predictive Maintenance System with MLOps Pipeline

> **Industrial IoT Analytics** • **Machine Learning** • **MLOps** • **Production Deployment**

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python) 
![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0-green?logo=fastapi) 
![Docker](https://img.shields.io/badge/Docker-24.0-blue?logo=docker) 
![AWS](https://img.shields.io/badge/AWS-ECS-orange?logo=amazon-aws) 
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2.5-blue?logo=githubactions)


A full-stack ML system that **predicts equipment failures before they occur**, built with **end-to-end MLOps practices**: experiment tracking, model versioning, containerized deployment, CI/CD automation, and cloud production readiness.

---

## 🎯 Key Achievements

* **91% peak accuracy** across multiple equipment types
* **<100ms prediction latency** for real-time inference
* **End-to-end pipeline**: data ingestion → preprocessing → model training → API serving → containerized deployment
* **Production-ready** FastAPI service with health checks, structured logging, and Pydantic validation
* **Automated CI/CD** with GitHub Actions for Docker build, ECR push, and ECS deployment
* **Scalable, cloud-ready** architecture using AWS ECR + ECS

---

## 📊 Project Overview

Predictive maintenance enables manufacturers to **preempt costly machine breakdowns**, improving operational efficiency and reducing downtime.
This system ingests **sensor data** (temperature, torque, speed, tool wear), trains ML models, and serves predictions via a **scalable API**, ready for production.

---

## 🛠 Technical Stack

**Machine Learning & Data:** scikit-learn, pandas, numpy, SMOTE
**API & Validation:** FastAPI, Pydantic
**MLOps & Experiment Tracking:** MLflow (experiments & model registry)
**Containerization:** Docker, Docker Compose
**CI/CD & Automation:** GitHub Actions (build → test → push → deploy)
**Cloud Deployment:** AWS ECR, ECS (production-ready, auto-scalable)
**Monitoring:** Structured logging, health endpoints, and API metrics

---

## 🏗 Architecture

```
Data → ETL → Model Training (MLflow) → Model Registry → API (FastAPI) 
→ Docker Container → GitHub Actions CI/CD → AWS ECR → ECS Production
```

**Highlights:**

* Fully **containerized ML service** for consistency across dev, staging, and prod
* Automated **CI/CD pipeline** ensures code & model changes are instantly deployed
* Cloud-ready deployment enables **horizontal scaling and high availability**

---

## 📸 Screenshots

![Capture](https://github.com/user-attachments/assets/b5ea1ab5-03d2-44e6-8820-187a19caeaef)

![Capture1](https://github.com/user-attachments/assets/f26830f7-deff-475a-bfeb-f62a10ef6608)

![Capture3](https://github.com/user-attachments/assets/31c9f394-42b5-43dd-a88c-2dd433af423d)

![Capture4](https://github.com/user-attachments/assets/d0772238-2de6-4264-ac18-958c9dec3bc4)

![Capture5](https://github.com/user-attachments/assets/9dbe47c1-21b9-4ef9-8144-6cafc4130858)

![Capture6](https://github.com/user-attachments/assets/6bc82e51-f6ae-40f0-b5e4-0aeee2da48ce) <img width="1878" height="693" alt="Screenshot 2025-08-15 120611" src="https://github.com/user-attachments/assets/afe3a0b4-9599-4a5a-8eb8-fd2dac558420" />

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

**Clone & Setup**

```bash
git clone https://github.com/Sa1f27/predictive-maintenance-mlops.git
cd predictive-maintenance-mlops

python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

**Run Pipeline**

```bash
# Train model
python run_pipeline.py --mode train

# Start API
python app.py
```

**Docker Deployment**

```bash
docker-compose up -d --build
```

**Production Deployment**

* GitHub Actions pipeline automatically builds Docker images, pushes to **AWS ECR**, and deploys to **ECS**
* Supports **zero-downtime updates** and **auto-scaling**

---

## 🔮 Future Enhancements

* **Advanced feature engineering:** rolling statistics, lag features
* **Ensemble/stacking models** for higher accuracy
* **Real-time data** with automated retraining
* **Monitoring & observability:** Prometheus/Grafana dashboards, alerting
* **Multi-region deployment** for global industrial clients
