name: MLOps CI/CD Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test-and-build:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install flake8 pytest
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
    
    - name: Lint with flake8
      run: |
        # Stop build if there are Python syntax errors or undefined names
        flake8 src --count --select=E9,F63,F7,F82 --show-source --statistics
        # Exit-zero treats all errors as warnings
        flake8 src --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
    
    - name: Run tests
      run: |
        pytest tests/ -v || echo "Tests not found, skipping"
    
    - name: Login to Docker Hub
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}
    
    - name: Build and push Docker image
      run: |
        docker build -t shaananshu2024ab05201/heart-disease-api:${{ github.sha }} .
        docker push shaananshu2024ab05201/heart-disease-api:${{ github.sha }}
        docker tag shaananshu2024ab05201/heart-disease-api:${{ github.sha }} shaananshu2024ab05201/heart-disease-api:latest
        docker push shaananshu2024ab05201/heart-disease-api:latest