"""
Heart Disease API
Shaan Anshu (2024AB05201)

API to serve our trained random forest model. The preprocessing was tricky -
only some features get scaled, not all of them.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from typing import List, Dict
import os
import logging
from datetime import datetime
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST, REGISTRY
from fastapi import Response
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Heart Disease Prediction API",
    description="MLOps Assignment",
    version="1.0.0"
)

MODEL_PATH = os.getenv('MODEL_PATH', '/app/models/random_forest_tuned.pkl')
PREPROCESSOR_PATH = os.getenv('PREPROCESSOR_PATH', '/app/models/preprocessor.pkl')

# Prometheus metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('api_request_duration_seconds', 'API request duration', ['method', 'endpoint'])
PREDICTION_COUNT = Counter('predictions_total', 'Total predictions made', ['prediction_label'])

# Load model and preprocessor
try:
    model = joblib.load(MODEL_PATH)
    prep_data = joblib.load(PREPROCESSOR_PATH)
    scaler = prep_data['scaler']
    feature_names = prep_data['feature_names']
    # Features that get scaled
    numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'hr_reserve']
    logger.info(f"Model loaded from {MODEL_PATH}")
    logger.info(f"Preprocessor loaded from {PREPROCESSOR_PATH}")
except Exception as e:
    logger.error(f"Failed to load files: {e}")
    model = None
    scaler = None
    feature_names = None


class PatientData(BaseModel):
    age: int = Field(..., ge=1, le=120)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=0, le=3)
    trestbps: int = Field(..., ge=50, le=250)
    chol: int = Field(..., ge=100, le=600)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: int = Field(..., ge=50, le=250)
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0, le=10)
    slope: int = Field(..., ge=0, le=2)
    ca: int = Field(..., ge=0, le=3)
    thal: int = Field(..., ge=0, le=3)
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
                "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
                "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
            }
        }


class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    confidence: float
    risk_level: str
    timestamp: str
    probabilities: Dict[str, float]


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
def home():
    return {
        "message": "Heart Disease Prediction API",
        "status": "running",
        "model_loaded": model is not None
    }


@app.get("/health")
def health():
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: PatientData):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to dataframe
        df = pd.DataFrame([data.model_dump()])
        
        # Add engineered features
        df['age_group'] = pd.cut(df['age'], bins=[0, 40, 55, 70, 120], labels=[0, 1, 2, 3]).astype(int)
        df['high_chol'] = (df['chol'] > 240).astype(int)
        df['high_bp'] = (df['trestbps'] > 140).astype(int)
        df['hr_reserve'] = 220 - df['age'] - df['thalach']
        
        # Scale ONLY the numerical features
        numerical_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'hr_reserve']
        X_numerical = df[numerical_to_scale]
        X_scaled = scaler.transform(X_numerical)
        
        # Put scaled values back
        for i, col in enumerate(numerical_to_scale):
            df[col] = X_scaled[0][i]
        
        # Now select all features in correct order
        X_final = df[feature_names]
        
        # Predict
        pred = model.predict(X_final)[0]
        probs = model.predict_proba(X_final)[0]
        conf = float(probs[pred])
        disease_prob = float(probs[1])
        
        # Determine risk level
        if disease_prob < 0.3:
            risk = "Low"
        elif disease_prob < 0.7:
            risk = "Medium"
        else:
            risk = "High"
        
        result = PredictionResponse(
            prediction=int(pred),
            prediction_label="Disease Present" if pred == 1 else "No Disease",
            confidence=round(conf, 4),
            risk_level=risk,
            timestamp=datetime.now().isoformat(),
            probabilities={
                "no_disease": round(float(probs[0]), 4),
                "disease": round(float(probs[1]), 4)
            }
        )
        
        logger.info(f"Prediction: {pred}, Confidence: {conf:.4f}")

        start_time = time.time()
        # Record metrics
        REQUEST_DURATION.labels(method='POST', endpoint='/predict').observe(time.time() - start_time)
        PREDICTION_COUNT.labels(prediction_label=result.prediction_label).inc()
        REQUEST_COUNT.labels(method='POST', endpoint='/predict', status=200).inc()

        return result
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(patients: List[PatientData]):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = []
        numerical_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'hr_reserve']
        
        for patient in patients:
            df = pd.DataFrame([patient.model_dump()])
            
            # Engineer features
            df['age_group'] = pd.cut(df['age'], bins=[0, 40, 55, 70, 120], labels=[0, 1, 2, 3]).astype(int)
            df['high_chol'] = (df['chol'] > 240).astype(int)
            df['high_bp'] = (df['trestbps'] > 140).astype(int)
            df['hr_reserve'] = 220 - df['age'] - df['thalach']
            
            # Scale numerical features
            X_numerical = df[numerical_to_scale]
            X_scaled = scaler.transform(X_numerical)
            for i, col in enumerate(numerical_to_scale):
                df[col] = X_scaled[0][i]
            
            X_final = df[feature_names]
            
            pred = model.predict(X_final)[0]
            probs = model.predict_proba(X_final)[0]
            
            results.append({
                "prediction": int(pred),
                "confidence": round(float(probs[pred]), 4),
                "probabilities": {
                    "no_disease": round(float(probs[0]), 4),
                    "disease": round(float(probs[1]), 4)
                }
            })
        
        return {"predictions": results, "count": len(results)}
        
    except Exception as e:
        logger.error(f"Batch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)