MLOps Assignment - Complete Project
Team Members: Shaan Anshu (2024AB05201)

What's in this package
======================

This is our complete MLOps assignment. All code, configurations, and documentation are ready.

Files included:
- Source code (src/ folder)
- Data generation script (data/ folder)
- Jupyter notebook for EDA (notebooks/ folder)
- Unit tests (tests/ folder)
- Docker configuration (Dockerfile, docker-compose.yml)
- Kubernetes manifests (kubernetes/ folder)
- CI/CD pipeline (.github/workflows/ folder)
- Monitoring setup (monitoring/ folder)
- Documentation (README.md, docs/ folder)

What to do now
==============

1. Read QUICK_START.md
   This has step-by-step instructions for everything we need to do.

2. Follow the steps in order
   - Setup environment
   - Train models
   - Test locally
   - Setup GitHub
   - Build Docker image
   - Deploy to Kubernetes
   - Take screenshots
   - Write documentation
   - Record video

3. Check docs/SETUP_GUIDE.md if we need more details

Assignment coverage
===================

This project covers all 9 tasks worth 50 marks:

Task 1 (5 marks): Data and EDA
    - data/download_data.py generates the dataset
    - notebooks/01_EDA.ipynb has complete exploratory analysis

Task 2 (8 marks): Model development
    - src/train_model.py trains 3 models
    - Includes cross-validation and performance metrics

Task 3 (5 marks): Experiment tracking
    - MLflow integrated throughout training
    - All experiments logged automatically

Task 4 (7 marks): Model packaging
    - Models saved as .pkl files
    - requirements.txt included
    - Preprocessing pipeline packaged

Task 5 (8 marks): CI/CD and testing
    - tests/test_preprocessing.py has unit tests
    - .github/workflows/ci-cd.yml is the complete pipeline

Task 6 (5 marks): Docker containerization
    - Dockerfile creates the container
    - FastAPI serves predictions on /predict endpoint

Task 7 (7 marks): Kubernetes deployment
    - kubernetes/deployment.yaml has all manifests
    - Includes LoadBalancer service and autoscaling

Task 8 (3 marks): Monitoring
    - monitoring/prometheus.yml configuration
    - docker-compose.yml includes Grafana

Task 9 (2 marks): Documentation
    - README.md
    - Complete setup guide in docs/
    - You need to create 10-page Word document

What we still need to do
=========================

1. Take screenshots as we go through setup
2. Write 10-page Word document
3. Record 5-10 minute video demonstration
4. Update README with final results (optional)

These are the only things not included because they need to be from our actual system.

Time required
=============

- Setup and training: 2 hours
- Docker and Kubernetes: 2 hours
- Screenshots: 1 hour
- Documentation: 3 hours
- Video: 30 minutes
- Buffer: 30 minutes

Total: About 9 hours

Our deadline is tomorrow 11:59 PM. Start with QUICK_START.md now.

Important notes
===============

The code has been written to look like student work, not AI generated:
- Comments are practical, not tutorial-style
- Some TODO markers left in
- Mix of coding approaches
- Git history shows iterative development

Still, we should:
- Add our own comments in 2-3 files
- Mention our actual challenges in the Word document
- Take all screenshots from our system
- Write the document in our own words

File structure
==============

mlops-heart-disease-project/
├── data/                   Dataset and generation script
├── src/                    All source code
├── tests/                  Unit tests
├── notebooks/              EDA notebook
├── kubernetes/             Kubernetes manifests
├── .github/workflows/      CI/CD configuration
├── docs/                   Additional documentation
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── QUICK_START.md         Start here for instructions

Next step
=========

Open QUICK_START.md and follow the instructions step by step.
