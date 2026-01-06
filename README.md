# Heart Disease Prediction - MLOps Assignment

**Team Members:** Shaan Anshu (2024AB05201)  
**Course:** MLOps (S1-25_AIMLCZG523)  
**Institution:** BITS Pilani

## Overview

This repository contains our MLOps assignment submission. The goal is to build a complete machine learning pipeline for predicting heart disease risk, including proper experiment tracking, containerization, CI/CD automation, and deployment on Kubernetes.

The dataset comes from UCI Machine Learning Repository - Heart Disease dataset with 303 patient records and 13 clinical features.

## What's included

The project covers all assignment requirements:
- Data preprocessing and exploratory analysis
- Multiple classification models with hyperparameter tuning
- MLflow for tracking experiments
- FastAPI service for model predictions
- Docker containers for deployment
- Kubernetes manifests for orchestration
- GitHub Actions for automated testing and deployment
- Basic monitoring setup with Prometheus

## Technical stack

- Python 3.9 with scikit-learn for ML
- MLflow for experiment tracking
- FastAPI for the REST API
- Docker for containerization
- Kubernetes (tested with Minikube)
- GitHub Actions for CI/CD
- Prometheus for monitoring

## Setup instructions

First, clone the repository and set up a Python virtual environment:

```bash
git clone https://github.com/ShaanAnshuBITS-2024AB05201/heart-disease-mlops.git
cd heart-disease-mlops
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Generate the dataset:

```bash
python data/download_data.py
```

Train the models:

```bash
cd src
python train_model.py
```

This trains three models - logistic regression, random forest, and a tuned random forest. The tuned version performs best with ROC-AUC around 0.90.

To view experiment results:

```bash
mlflow ui
```

Then open http://localhost:5000 in your browser.

## Running the API

Start the FastAPI server:

```bash
python src/app.py
```

The API will be available at http://localhost:8000. Interactive documentation is at http://localhost:8000/docs.

Example prediction request:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1}'
```

## Docker deployment

Build the Docker image:

```bash
docker build -t shaananshu2024ab05201/heart-disease-api:latest .
```

Run the container:

```bash
docker run -d -p 8000:8000 shaananshu2024ab05201/heart-disease-api:latest
```

Push to Docker Hub (after docker login):

```bash
docker push shaananshu2024ab05201/heart-disease-api:latest
```

## Kubernetes deployment

Start Minikube and deploy:

```bash
minikube start
kubectl apply -f kubernetes/deployment.yaml
```

Access the service:

```bash
minikube tunnel  # Run this in a separate terminal
kubectl get services
```

The API will be accessible through the external IP shown by kubectl.

## Testing

Run unit tests:

```bash
pytest tests/ -v
```

For coverage report:

```bash
pytest tests/ --cov=src --cov-report=html
```

## CI/CD pipeline

The GitHub Actions workflow runs automatically on push to main. It handles:
- Code linting with flake8
- Unit tests with pytest
- Model training
- Docker image build and push

To set this up, add these secrets to your GitHub repository:
- DOCKER_USERNAME: shaananshu2024ab05201
- DOCKER_PASSWORD: your Docker Hub password

## Project structure

```
.
├── data/                   # Dataset and generation script
├── src/                    # Source code
│   ├── preprocessing.py    # Data preprocessing
│   ├── train_model.py     # Model training
│   └── app.py             # FastAPI application
├── tests/                  # Unit tests
├── notebooks/              # Jupyter notebooks for EDA
├── kubernetes/             # Kubernetes manifests
├── .github/workflows/      # CI/CD configuration
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Model performance

After training and tuning, these are the test set results:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.8525 | 0.8182 | 0.8571 | 0.8372 | 0.8893 |
| Random Forest | 0.8689 | 0.8462 | 0.8571 | 0.8516 | 0.8982 |
| Tuned Random Forest | 0.8852 | 0.8636 | 0.9048 | 0.8837 | 0.9045 |

The tuned random forest is used in production deployment.

## Notes

A few things we learned while working on this:

- MLflow tracking requires the mlruns directory to exist and have proper permissions. Had some issues initially with artifact logging.
- The Docker image size was large at first (over 2GB). Using python:3.9-slim as base reduced it significantly.
- For Kubernetes on Minikube, you need to run `minikube tunnel` to access LoadBalancer services.
- GitHub Actions needs the DOCKER_USERNAME and DOCKER_PASSWORD secrets configured or the pipeline will fail at the push step.

## Assignment requirements mapping

- Task 1 (Data & EDA): See notebooks/01_EDA.ipynb and screenshots folder
- Task 2 (Model Development): Three models in src/train_model.py with cross-validation
- Task 3 (Experiment Tracking): MLflow integration throughout training pipeline
- Task 4 (Model Packaging): Models saved as .pkl files with preprocessing pipeline
- Task 5 (CI/CD): GitHub Actions workflow in .github/workflows/ci-cd.yml
- Task 6 (Containerization): Dockerfile with FastAPI application
- Task 7 (Kubernetes): Deployment manifests in kubernetes/ folder
- Task 8 (Monitoring): Prometheus configuration in monitoring/
- Task 9 (Documentation): This README and additional docs in docs/ folder

## Contact

Team Lead: Shaan Anshu  
Email: 2024ab05201@wilp.bits-pilani.ac.in  
BITS ID: 2024AB05201

---



---


