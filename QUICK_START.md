QUICK START GUIDE

Team Members: Shaan Anshu (2024AB05201)
Assignment: MLOps Heart Disease Prediction

What we need to do
===================

STEP 1: Setup environment (15 minutes)

Extract the project folder and open terminal inside it.

Create virtual environment:
    python -m venv venv
    
Activate it:
    Windows: venv\Scripts\activate
    Mac/Linux: source venv/bin/activate

Install dependencies:
    pip install -r requirements.txt

STEP 2: Generate data and train models (15 minutes)

Generate the dataset:
    python data/download_data.py

Train models:
    cd src
    python train_model.py
    
This trains three models and logs everything to MLflow. Takes about 10 minutes.

STEP 3: Test locally (10 minutes)

Run the API:
    python src/app.py

In another terminal, test it:
    curl http://localhost:8000/health

Or open http://localhost:8000/docs in your browser for interactive testing.

STEP 4: Setup GitHub (10 minutes)

Create a new repository on GitHub:
    - Name: heart-disease-mlops
    - Public repository
    - Don't initialize with README

Initialize git with realistic commits:
    bash init_git.sh

Add your repository:
    git remote add origin https://github.com/ShaanAnshuBITS-2024AB05201/heart-disease-mlops.git
    git push -u origin main

STEP 5: Setup GitHub Actions (5 minutes)

Go to your repository on GitHub.
Click Settings > Secrets and variables > Actions
Add two secrets:
    - DOCKER_USERNAME: shaananshu2024ab05201
    - DOCKER_PASSWORD: (your docker hub password)

Now every push will trigger the CI/CD pipeline.

STEP 6: Build and push Docker image (20 minutes)

Login to Docker Hub:
    docker login

Build the image:
    docker build -t shaananshu2024ab05201/heart-disease-api:latest .

Test it locally:
    docker run -d -p 8000:8000 shaananshu2024ab05201/heart-disease-api:latest
    curl http://localhost:8000/health

Push to Docker Hub:
    docker push shaananshu2024ab05201/heart-disease-api:latest

STEP 7: Deploy to Kubernetes (20 minutes)

Start Minikube:
    minikube start
    minikube addons enable metrics-server

Deploy the application:
    kubectl apply -f kubernetes/deployment.yaml

Check if pods are running:
    kubectl get pods

This might take 2-3 minutes. Wait until pods show "Running" status.

Access the service:
    minikube tunnel
    
In another terminal:
    kubectl port-forward service/heart-disease-api-service 8080:80

Test:
    curl http://localhost:8080/health

STEP 8: Take screenshots (30 minutes)

We need screenshots for documentation. Capture these:

1. MLflow UI showing experiments (run: mlflow ui)
2. Jupyter notebook with EDA visualizations
3. Docker build output
4. Docker containers running (docker ps)
5. API documentation at http://localhost:8000/docs
6. Successful API prediction response
7. Kubernetes pods (kubectl get pods)
8. Kubernetes services (kubectl get services)
9. GitHub Actions workflow (green checkmarks)
10. Prometheus dashboard (if using docker-compose)

Save all screenshots in a screenshots/ folder.

STEP 9: Write 10-page document (3 hours)

Structure the Word document like this:

Page 1: Title page
- Project title
- Team members and BITS IDs
- Course details
- Date

Page 2-3: Dataset and EDA
- Dataset description
- Our EDA process and findings
- Include visualizations from notebook

Page 4-5: Model development
- Models we tried (Logistic Regression, Random Forest, Tuned RF)
- Our hyperparameter tuning approach
- Performance comparison table
- MLflow screenshots

Page 6: Containerization
- Our Docker setup
- Dockerfile breakdown
- Testing approach

Page 7: CI/CD
- GitHub Actions workflow explanation
- Pipeline stages
- Screenshots of successful runs

Page 8: Kubernetes deployment
- Deployment configuration
- Service setup
- Scaling configuration
- Screenshots

Page 9: Monitoring
- Prometheus setup
- Our monitoring approach
- Screenshots

Page 10: Conclusion
- What we learned
- Challenges we faced
- Possible improvements

STEP 10: Record video (20 minutes)

Record screen while demonstrating:
1. Training models
2. MLflow UI
3. Docker build and run
4. API testing
5. Kubernetes deployment
6. Testing deployed service

Keep it between 5-10 minutes. Show the working system.

Common issues
=============

Docker build fails:
    Make sure models exist in models/ folder.
    If not, run: cd src && python train_model.py

Kubernetes pods not starting:
    Check: kubectl describe pod <pod-name>
    Usually image pull issue or wrong image name.

MLflow shows no experiments:
    Make sure you ran python src/train_model.py
    Check that mlruns/ folder exists.

API returns 503:
    Models not loaded. Train them first with train_model.py

Time estimate
=============

Today:
- Setup and training: 2 hours
- Docker and Kubernetes: 2 hours
- Screenshots: 1 hour

Tomorrow:
- Documentation: 3 hours
- Video: 30 minutes
- Final review: 30 minutes

Total: About 9 hours spread over 2 days.

We have until tomorrow 11:59 PM. Plenty of time if we start now.

Contact
=======

If something breaks, check:
- Error messages carefully
- Docker/Kubernetes logs
- GitHub Actions logs

The docs/SETUP_GUIDE.md file has more detailed troubleshooting.
