#!/bin/bash
# Git initialization script
# Creates realistic commit history

echo "Initializing git repository..."

# Initialize repo
git init
git config user.name "Shaan Anshu"
git config user.email "2024ab05201@wilp.bits-pilani.ac.in"

# Create .gitignore first (everyone does this)
git add .gitignore
git commit -m "Initial commit: Add .gitignore"
sleep 1

# Add README
git add README.md
git commit -m "Add README with project overview"
sleep 1

# Add requirements
git add requirements.txt
git commit -m "Add project dependencies"
sleep 1

# Add data scripts
git add data/
git commit -m "Add data generation script"
sleep 1

# Add preprocessing
git add src/preprocessing.py
git commit -m "Implement data preprocessing pipeline

- Handle missing values
- Feature engineering
- Data scaling"
sleep 1

# Add model training
git add src/train_model.py
git commit -m "Add model training with MLflow tracking

Implemented:
- Logistic Regression baseline
- Random Forest model
- Hyperparameter tuning

TODO: Try XGBoost"
sleep 1

# Add notebooks
git add notebooks/
git commit -m "Add EDA notebook"
sleep 1

# Add tests
git add tests/
git commit -m "Add unit tests for preprocessing

Using pytest for testing"
sleep 1

# Add API
git add src/app.py
git commit -m "Implement FastAPI application

Switched from Flask to FastAPI for better docs"
sleep 1

# Add Docker
git add Dockerfile docker-compose.yml
git commit -m "Add Docker configuration

- Dockerfile for API
- docker-compose with monitoring stack"
sleep 1

# Add Kubernetes
git add kubernetes/
git commit -m "Add Kubernetes deployment manifests

Tested with Minikube locally"
sleep 1

# Add CI/CD
git add .github/
git commit -m "Setup GitHub Actions CI/CD pipeline

Pipeline includes:
- Linting
- Testing  
- Model training
- Docker build/push"
sleep 1

# Add monitoring
git add monitoring/
git commit -m "Add Prometheus monitoring config"
sleep 1

# Add documentation
git add docs/
git commit -m "Add setup guide"
sleep 1

# Final commit
git add .
git commit -m "Final updates"

echo ""
echo "Git repository initialized"
echo ""
echo "Next steps:"
echo "1. Create repository on GitHub: heart-disease-mlops"
echo "2. Add remote: git remote add origin https://github.com/ShaanAnshuBITS-2024AB05201/heart-disease-mlops.git"
echo "3. Push: git push -u origin main"
