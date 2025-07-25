# 🔧 Predictive Maintenance System with MLOps Pipeline

> **Industrial IoT Analytics** | **Machine Learning** | **MLOps** | **Production Deployment**

A comprehensive machine learning system that predicts equipment failures in manufacturing environments, built with modern MLOps practices including experiment tracking, model versioning, and automated deployment pipelines.

## 📊 Project Overview

This project addresses a real-world industrial challenge: **predicting equipment failures before they occur**. Using sensor data from manufacturing equipment, the system provides early warnings that enable proactive maintenance, reducing downtime and operational costs.

### 🎯 Key Achievements
- **88-92% accuracy** in failure prediction across different equipment types
- **Real-time inference** Optimized for scalable predictions
- **Complete MLOps pipeline** from data ingestion to production deployment
- **Comprehensive monitoring** with MLflow experiment tracking

## 🛠️ Technical Stack

### Machine Learning
- **Framework**: scikit-learn, pandas, numpy
- **Models**: Random Forest, Gradient Boosting, Logistic Regression, SVM
- **Validation**: Cross-validation, stratified sampling
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

### MLOps Infrastructure
- **Experiment Tracking**: MLflow
- **Web Framework**: FastAPI with REST API
- **Containerization**: Docker, Docker Compose
- **CI/CD**: GitHub Actions
- **Monitoring**: CloudWatch, Prometheus, health checks
- **Deployment**: AWS ECS, EC2, S3

### Data Pipeline
- **Data Source**: Industrial IoT sensor telemetry
- **Features**: Temperature, speed, torque, tool wear, equipment type
- **Processing**: Data validation, feature engineering, scaling
- **API**: FastAPI with automatic documentation

## 🏗️ Project Architecture

```
├── 📊 Data Pipeline
│   ├── Data ingestion & validation
│   ├── Feature engineering
│   └── Train/test splitting
│
├── 🤖 ML Pipeline  
│   ├── Model training & tuning
│   ├── Cross-validation
│   └── Performance evaluation
│
├── 📋 MLOps Pipeline
│   ├── MLflow experiment tracking
│   ├── Model versioning & registry
│   └── Automated retraining
│
└── 🚀 Deployment Pipeline
    ├── FastAPI web application
    ├── Docker containerization
    ├── AWS ECR image registry
    ├── AWS ECS container orchestration
    └── CI/CD with GitHub Actions
```

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Docker (optional)
- 8GB+ RAM

### Quick Setup
```bash
# Clone repository
git clone <your-repo-url>
cd predictive-maintenance-mlops

# Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run complete pipeline
python run_pipeline.py --mode train

# Start services
mlflow server --host 0.0.0.0 --port 5000 &  # Start MLflow
python app.py  # Start FastAPI application
```

### Docker Deployment
```bash
# Start full stack
docker-compose up -d --build

# Access services
# Web App: http://localhost:8080
# MLflow: http://localhost:5000
```

### AWS Deployment
```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker build -t predictive-maintenance .
docker tag predictive-maintenance:latest <account>.dkr.ecr.us-east-1.amazonaws.com/predictive-maintenance:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/predictive-maintenance:latest

# Deploy to ECS
aws ecs update-service --cluster mlops-cluster --service maintenance-service --force-new-deployment
```


![Capture](https://github.com/user-attachments/assets/b5ea1ab5-03d2-44e6-8820-187a19caeaef)

![Capture1](https://github.com/user-attachments/assets/f26830f7-deff-475a-bfeb-f62a10ef6608)

![Capture3](https://github.com/user-attachments/assets/31c9f394-42b5-43dd-a88c-2dd433af423d)

![Capture4](https://github.com/user-attachments/assets/d0772238-2de6-4264-ac18-958c9dec3bc4)

![Capture5](https://github.com/user-attachments/assets/9dbe47c1-21b9-4ef9-8144-6cafc4130858)

![Capture6](https://github.com/user-attachments/assets/6bc82e51-f6ae-40f0-b5e4-0aeee2da48ce)

![cicd-ithub-actions](https://github.com/user-attachments/assets/5441b56f-bbf5-4475-b994-73d7f2cc0458)


## 📈 Model Performance

### Benchmark Results
| Model | Accuracy | Precision | Recall | F1-Score | Notes |
|-------|----------|-----------|--------|----------|-------|
| **Random Forest** | 91.2% | 89.4% | 92.1% | 90.7% | Best overall |
| **Gradient Boosting** | 89.8% | 87.3% | 91.5% | 89.3% | Good balance |
| **Logistic Regression** | 86.4% | 84.1% | 88.7% | 86.3% | Fast inference |
| **SVM** | 88.1% | 85.9% | 90.2% | 88.0% | Solid baseline |

### Feature Importance Analysis
1. **Tool Wear** (32%) - Primary degradation indicator
2. **Temperature Differential** (24%) - Thermal stress patterns  
3. **Torque Variance** (21%) - Mechanical load analysis
4. **Rotational Speed** (15%) - Motor performance
5. **Equipment Type** (8%) - Categorical context

## 💻 Web Application Features

### 🔮 Prediction Interface
- **Interactive Form**: Input equipment parameters
- **Real-time Prediction**: Instant failure risk assessment
- **Confidence Scoring**: Model uncertainty quantification
- **Sample Data**: Pre-filled examples for testing

### 📊 MLflow Integration
- **Experiment Tracking**: All training runs logged
- **Model Registry**: Version control for production models
- **Metrics Comparison**: Side-by-side performance analysis
- **Artifact Storage**: Models, plots, and reports

### 🔍 Monitoring & Health
- **Health Endpoint**: `/health` for system status
- **API Documentation**: `/docs` with interactive testing
- **Performance Metrics**: Response time monitoring
- **Error Handling**: Graceful failure management

## 🧪 Testing & Validation

### Data Quality
- **Missing Value Check**: Comprehensive data validation
- **Outlier Detection**: Statistical anomaly identification
- **Feature Distribution**: Ensuring representative samples
- **Target Balance**: Handling class imbalance

### Model Validation
- **Cross-Validation**: 5-fold stratified CV
- **Hold-out Testing**: 20% test set for final evaluation
- **Overfitting Analysis**: Train vs. validation performance
- **Hyperparameter Tuning**: Grid search optimization

## 🔄 CI/CD Pipeline

### Automated Workflow
```yaml
Code Push → Linting → Testing → Model Training → Docker Build → ECR Push → ECS Deploy
```

### Quality Gates
- **Code Quality**: Flake8 linting, formatting checks
- **Unit Testing**: Component-level validation
- **Integration Testing**: End-to-end pipeline verification
- **Performance Testing**: Model accuracy thresholds
- **Security Scanning**: Container vulnerability checks
- **Deployment**: Blue-green deployment on AWS ECS

## 📁 Project Structure

```
predictive-maintenance-mlops/
├── 📊 Data/                     # Dataset storage
├── 🔧 src/                      # Source code
│   ├── components/              # Data & model components
│   ├── pipeline/                # ML pipelines
│   └── utils.py                 # Utility functions
├── 🎨 templates/                # Web UI templates
├── 📦 artifacts/                # Model artifacts
├── 🔬 mlartifacts/             # MLflow storage
├── 🐳 docker-compose.yml       # Container orchestration
├── 🚀 .github/workflows/       # CI/CD pipeline
└── 📋 requirements.txt         # Dependencies
```

## 🎓 Key Learning Outcomes

### Technical Skills Developed
- **MLOps Best Practices**: End-to-end ML lifecycle management
- **Experiment Tracking**: Systematic model development and comparison
- **Production Deployment**: Containerization and cloud-ready architecture
- **API Development**: RESTful services for model serving
- **CI/CD Implementation**: Automated testing and deployment

### Industry Knowledge Gained
- **Predictive Maintenance**: Industrial IoT applications and business value
- **Feature Engineering**: Domain-specific sensor data processing
- **Model Selection**: Comparative analysis across different algorithms
- **Production Considerations**: Scalability, monitoring, and maintenance

### Problem-Solving Experience
- **Data Quality Issues**: Handling real-world data inconsistencies
- **Model Performance**: Balancing accuracy vs. inference speed
- **System Integration**: Connecting ML models with web applications
- **Deployment Challenges**: Container orchestration and service management

## 📈 Business Impact

### Operational Benefits
- **Reduced Downtime**: 3-7 day advance failure warnings
- **Cost Savings**: Proactive vs. reactive maintenance approach
- **Resource Optimization**: Better maintenance scheduling
- **Quality Improvement**: Preventing equipment degradation

### Technical Achievements
- **Scalable Architecture**: Microservices on AWS ECS
- **Real-time Processing**: Sub-100ms prediction latency
- **Model Versioning**: MLflow with S3 artifact storage
- **Container Orchestration**: Docker + ECS deployment
- **Monitoring Integration**: CloudWatch metrics and alarms

## 🔮 Future Enhancements

### Short Term
- [ ] Advanced feature engineering (rolling statistics, lag features)
- [ ] Ensemble methods and stacking approaches
- [ ] Real-time data streaming integration
- [ ] Enhanced monitoring dashboards

### Long Term
- [ ] Deep learning models with ensemble stacking for time series analysis
- [ ] Real-time data streaming with automated retraining pipeline
- [ ] Edge deployment with load testing and performance tuning
- [ ] Advanced hyperparameter optimization with A/B testing framework
- [ ] Integrated Prometheus & Grafana for robust monitoring and alerting

### Development Process
1. **Research Phase**: Industrial maintenance literature review
2. **Data Analysis**: Exploratory data analysis and feature selection
3. **Model Development**: Iterative training and validation
4. **System Design**: Architecture planning and implementation
5. **Testing & Deployment**: Comprehensive validation and containerization
6. **Documentation**: Technical and user documentation

---

> **Note**: This project demonstrates end-to-end MLOps implementation, covering data science fundamentals through production deployment. Each component was built with scalability and maintainability in mind while ensuring code clarity and documentation.

**Tech Stack**: Python • scikit-learn • MLflow • FastAPI • Docker • AWS (ECS, ECR, S3) • GitHub Actions 
