Setup and Deployment Guide

Team Members: Shaan Anshu (2024AB05201)
Date: January 2026

Prerequisites
=============

Software needed:
- Python 3.9 or higher
- Docker Desktop with Kubernetes enabled
- Git
- kubectl (comes with Docker Desktop)

Accounts needed:
- GitHub account
- Docker Hub account

Verify installations:
    python --version
    docker --version
    kubectl version --client
    git --version

Initial Setup
=============

1. Clone repository

    git clone https://github.com/ShaanAnshuBITS-2024AB05201/heart-disease-mlops.git
    cd heart-disease-mlops

2. Create virtual environment

    python -m venv venv
    
    On Windows:
        venv\Scripts\activate
    
    On Mac/Linux:
        source venv/bin/activate

3. Install dependencies

    pip install --upgrade pip
    pip install -r requirements.txt

Local Development
=================

Generate dataset:

    python data/download_data.py

Expected output: "Dataset generated: (303, 14)"

Train models:

    cd src
    python train_model.py

This takes about 10 minutes. Trains three models and logs to MLflow.

View experiments:

    mlflow ui

Open http://localhost:5000 in browser.

Run API:

    python src/app.py

Test it:

    curl http://localhost:8000/health

Or open http://localhost:8000/docs for interactive testing.

Docker Deployment
=================

Build image:

    docker build -t shaananshu2024ab05201/heart-disease-api:latest .

Run container:

    docker run -d -p 8000:8000 --name heart-api \
      shaananshu2024ab05201/heart-disease-api:latest

Check logs:

    docker logs heart-api

Test:

    curl http://localhost:8000/health

Push to Docker Hub:

    docker login
    docker push shaananshu2024ab05201/heart-disease-api:latest

Kubernetes Deployment
======================

Start Minikube:

    minikube start
    minikube addons enable metrics-server
    kubectl get nodes

Deploy application:

    kubectl apply -f kubernetes/deployment.yaml

Check status:

    kubectl get deployments
    kubectl get pods
    kubectl get services

Wait for pods to show "Running" status (takes 2-3 minutes).

Access service:

In one terminal:
    minikube tunnel

In another terminal:
    kubectl port-forward service/heart-disease-api-service 8080:80

Test:

    curl http://localhost:8080/health

Make a prediction:

    curl -X POST "http://localhost:8080/predict" \
      -H "Content-Type: application/json" \
      -d '{"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1}'

Scale deployment:

    kubectl scale deployment heart-disease-api --replicas=3
    kubectl get pods

Check autoscaling:

    kubectl get hpa

View logs:

    kubectl get pods
    kubectl logs <pod-name>
    kubectl logs -f <pod-name>  # Follow logs

CI/CD Setup
===========

Add GitHub secrets:

1. Go to repository on GitHub
2. Settings > Secrets and variables > Actions
3. Add two secrets:
   - DOCKER_USERNAME: shaananshu2024ab05201
   - DOCKER_PASSWORD: (your docker hub password)

Push code:

    git add .
    git commit -m "Initial commit"
    git push origin main

Check pipeline:

Go to Actions tab on GitHub. Pipeline should run automatically.

Pipeline stages:
1. Lint and test
2. Build and train model
3. Build Docker image
4. Push to Docker Hub (only on main branch)

Monitoring Setup
================

Using docker-compose:

    docker-compose up -d

Access dashboards:
- API: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000
  Login: admin / admin123

Grafana setup:

1. Open http://localhost:3000
2. Login with admin/admin123
3. Add data source:
   - Type: Prometheus
   - URL: http://prometheus:9090
   - Click "Save & Test"
4. Import dashboard:
   - Dashboard ID: 1860
   - Select Prometheus data source

Troubleshooting
===============

Docker build fails:

Check if models exist:
    ls models/

If missing, train first:
    cd src && python train_model.py

Kubernetes pods not starting:

Check pod details:
    kubectl describe pod <pod-name>

Common issues:
- Image pull error: Check Docker Hub image exists
- Resource limits: Reduce limits in deployment.yaml

MLflow shows no experiments:

Check mlruns directory:
    ls mlruns/

If missing, retrain:
    cd src && python train_model.py

API returns 503:

Models not loaded. Train them:
    cd src && python train_model.py

Check model files exist:
    ls models/*.pkl

Minikube tunnel asks for password repeatedly:

Use port-forward instead:
    kubectl port-forward service/heart-disease-api-service 8080:80

Screenshots for Assignment
===========================

Capture these:

1. EDA visualizations from Jupyter notebook
2. MLflow UI showing experiments
3. Model comparison table
4. Docker build output
5. Docker ps showing running container
6. API documentation (http://localhost:8000/docs)
7. Successful prediction response
8. kubectl get pods
9. kubectl get services
10. GitHub Actions workflow (green checks)
11. Prometheus dashboard
12. Grafana dashboard

Save all in screenshots/ folder.

Final Checklist
===============

Before submission:

- Code pushed to GitHub
- Docker image on Docker Hub  
- CI/CD pipeline passing
- All screenshots captured
- 10-page Word document complete
- Video recorded
- Everything tested end-to-end

Contact
=======

If issues not covered here:
- Check error messages
- Check Docker logs: docker logs <container>
- Check Kubernetes logs: kubectl logs <pod>
- Check GitHub Actions logs on Actions tab
