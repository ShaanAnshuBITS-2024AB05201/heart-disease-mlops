"""
Model training script with MLflow experiment tracking
Team Members: Shaan Anshu (2024AB05201)
Date: January 2026

This script trains multiple models for heart disease prediction and tracks
experiments using MLflow. After several iterations, Random Forest with 
hyperparameter tuning gives the best results.

TODO: Experiment with XGBoost if time permits
TODO: Add early stopping for RF training
"""
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys
import warnings
import tempfile
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.dirname(__file__))
from preprocessing import prepare_train_test_data


class HeartDiseaseModelTrainer:
    """
    Model trainer class for heart disease prediction
    
    Initially tried just Logistic Regression but Random Forest performed
    significantly better. Hyperparameter tuning improved results by ~2-3%.
    """
    
    def __init__(self, experiment_name="heart-disease-prediction"):
        """Initialize model trainer"""
        self.experiment_name = experiment_name
        self.model = None
        self.model_name = None
        self.current_run = None
        self.artifact_dir = tempfile.mkdtemp()  # Create temp directory for artifacts
        
        # Set MLflow tracking
        mlflow.set_experiment(experiment_name)
        print(f"MLflow experiment set: {experiment_name}")
        print(f"Artifact directory: {self.artifact_dir}")
        
    def train_logistic_regression(self, X_train, y_train, params=None):
        """Train Logistic Regression model"""
        if params is None:
            params = {
                'C': 1.0,
                'max_iter': 1000,
                'random_state': 42,
                'solver': 'lbfgs'
            }
        
        self.current_run = mlflow.start_run(run_name="Logistic_Regression")
        with self.current_run:
            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param("model_type", "Logistic Regression")
            
            # Train model
            self.model = LogisticRegression(**params)
            self.model.fit(X_train, y_train)
            self.model_name = "logistic_regression"
            
            # Cross-validation
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='roc_auc')
            mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
            mlflow.log_metric("cv_roc_auc_std", cv_scores.std())
            
            print(f"Logistic Regression trained")
            print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            
            return self.model
    
    def train_random_forest(self, X_train, y_train, params=None):
        """Train Random Forest model"""
        if params is None:
            params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1
            }
        
        self.current_run = mlflow.start_run(run_name="Random_Forest")
        with self.current_run:
            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param("model_type", "Random Forest")
            
            # Train model
            self.model = RandomForestClassifier(**params)
            self.model.fit(X_train, y_train)
            self.model_name = "random_forest"
            
            # Cross-validation
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='roc_auc')
            mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
            mlflow.log_metric("cv_roc_auc_std", cv_scores.std())
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Save feature importance plot
            plt.figure(figsize=(10, 6))
            sns.barplot(data=feature_importance.head(10), x='importance', y='feature', palette='viridis')
            plt.title('Top 10 Feature Importances')
            plt.xlabel('Importance')
            plt.tight_layout()
            
            # Save to temp directory
            importance_path = os.path.join(self.artifact_dir, 'feature_importance.png')
            plt.savefig(importance_path, dpi=300, bbox_inches='tight')
            mlflow.log_artifact(importance_path)
            plt.close()
            print(f"Feature importance plot saved and logged")
            
            print(f"Random Forest trained")
            print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            
            return self.model
    
    def tune_hyperparameters(self, X_train, y_train, model_type='random_forest'):
        """Hyperparameter tuning with GridSearchCV"""
        print(f"Tuning {model_type}...")
        
        if model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        else:
            param_grid = {
                'C': [0.01, 0.1, 1.0, 10.0],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'liblinear']
            }
            base_model = LogisticRegression(max_iter=1000, random_state=42)
        
        self.current_run = mlflow.start_run(run_name=f"{model_type}_tuned")
        with self.current_run:
            grid_search = GridSearchCV(
                base_model, param_grid, cv=5, scoring='roc_auc', 
                n_jobs=-1, verbose=0
            )
            grid_search.fit(X_train, y_train)
            
            # Log best parameters
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_param("model_type", f"{model_type}_tuned")
            mlflow.log_metric("best_cv_score", grid_search.best_score_)
            
            self.model = grid_search.best_estimator_
            self.model_name = f"{model_type}_tuned"
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
            
            return self.model
    
    def evaluate(self, X_test, y_test, save_plots=True):
        """Evaluate model and log metrics to MLflow"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        print(f"Evaluating {self.model_name}...")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Log metrics
        if self.current_run:
            with self.current_run:
                mlflow.log_metric("test_accuracy", accuracy)
                mlflow.log_metric("test_precision", precision)
                mlflow.log_metric("test_recall", recall)
                mlflow.log_metric("test_f1_score", f1)
                mlflow.log_metric("test_roc_auc", roc_auc)
                
                # Print metrics
                print(f"Accuracy:  {accuracy:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall:    {recall:.4f}")
                print(f"F1-Score:  {f1:.4f}")
                print(f"ROC-AUC:   {roc_auc:.4f}")
                
                # Confusion Matrix
                if save_plots:
                    cm = confusion_matrix(y_test, y_pred)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
                    plt.title('Confusion Matrix')
                    plt.ylabel('Actual')
                    plt.xlabel('Predicted')
                    plt.tight_layout()
                    
                    cm_path = os.path.join(self.artifact_dir, f'confusion_matrix_{self.model_name}.png')
                    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
                    mlflow.log_artifact(cm_path)
                    plt.close()
                    print(f"Confusion matrix saved and logged")
                
                # ROC Curve
                if save_plots:
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
                    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('ROC Curve')
                    plt.legend()
                    plt.grid(alpha=0.3)
                    plt.tight_layout()
                    
                    roc_path = os.path.join(self.artifact_dir, f'roc_curve_{self.model_name}.png')
                    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
                    mlflow.log_artifact(roc_path)
                    plt.close()
                    print(f"ROC curve saved and logged")
                
                # Classification Report
                report = classification_report(y_test, y_pred)
                print(f"\nClassification Report:\n{report}")
                
                # Save classification report
                if save_plots:
                    report_path = os.path.join(self.artifact_dir, f'classification_report_{self.model_name}.txt')
                    with open(report_path, 'w') as f:
                        f.write(report)
                    mlflow.log_artifact(report_path)
                    print(f"Classification report saved and logged")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
    
    def save_model(self, model_dir='../models'):
        """Save model using MLflow and pickle"""
        if self.model is None:
            raise ValueError("No model to save!")
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save with MLflow
        if self.current_run:
            with self.current_run:
                mlflow.sklearn.log_model(self.model, "model")
                print(f"Model logged to MLflow")
        
        # Save with pickle
        model_path = os.path.join(model_dir, f'{self.model_name}.pkl')
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")
        
        # End current run
        if self.current_run:
            mlflow.end_run()
            self.current_run = None
        
        return model_path


def main():
    """Main training pipeline"""
    print("Starting model training pipeline...")
    
    # Load and prepare data
    print("\nLoading and preprocessing data...")
    data_path = '../data/heart_disease_raw.csv'
    X_train, X_test, y_train, y_test, preprocessor = prepare_train_test_data(data_path)
    
    # Save preprocessor
    preprocessor.save('../models/preprocessor.pkl')
    
    # Initialize trainer
    trainer = HeartDiseaseModelTrainer()
    
    # Train models
    print("\nTraining models...")
    
    # Model 1: Logistic Regression
    trainer.train_logistic_regression(X_train, y_train)
    metrics_lr = trainer.evaluate(X_test, y_test)
    trainer.save_model()
    
    # Model 2: Random Forest
    trainer.train_random_forest(X_train, y_train)
    metrics_rf = trainer.evaluate(X_test, y_test)
    trainer.save_model()
    
    # Model 3: Tuned Random Forest (Best model)
    print("\nHyperparameter tuning...")
    trainer.tune_hyperparameters(X_train, y_train, model_type='random_forest')
    metrics_tuned = trainer.evaluate(X_test, y_test)
    trainer.save_model()
    
    # Compare models
    print("\nModel Comparison:")
    comparison = pd.DataFrame({
        'Model': ['Logistic Regression', 'Random Forest', 'Tuned Random Forest'],
        'Accuracy': [metrics_lr['accuracy'], metrics_rf['accuracy'], metrics_tuned['accuracy']],
        'Precision': [metrics_lr['precision'], metrics_rf['precision'], metrics_tuned['precision']],
        'Recall': [metrics_lr['recall'], metrics_rf['recall'], metrics_tuned['recall']],
        'F1-Score': [metrics_lr['f1_score'], metrics_rf['f1_score'], metrics_tuned['f1_score']],
        'ROC-AUC': [metrics_lr['roc_auc'], metrics_rf['roc_auc'], metrics_tuned['roc_auc']]
    })
    print(comparison.to_string(index=False))
    
    # Save comparison
    comparison.to_csv('../models/model_comparison.csv', index=False)
    
    print("\nTraining complete!")
    print(f"Models saved in: ../models/")
    print(f"MLflow UI: Run 'mlflow ui' to view experiments")
    print(f"Best model: Tuned Random Forest (ROC-AUC: {metrics_tuned['roc_auc']:.4f})")


if __name__ == "__main__":
    main()