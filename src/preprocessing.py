"""
Data preprocessing pipeline for Heart Disease prediction
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os


class DataPreprocessor:
    """
    Preprocessor for Heart Disease dataset
    Handles missing values, encoding, and scaling
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = None
        self.numerical_features = [
            'age', 'trestbps', 'chol', 'thalach', 'oldpeak'
        ]
        self.categorical_features = [
            'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'
        ]
        
    def load_data(self, filepath):
        """Load dataset from CSV"""
        df = pd.read_csv(filepath)
        print(f"Data loaded: {df.shape}")
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values using median/mode imputation"""
        df = df.copy()
        
        # Numerical features: fill with median
        for col in self.numerical_features:
            if col in df.columns and df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
        
        # Categorical features: fill with mode
        for col in self.categorical_features:
            if col in df.columns and df[col].isnull().any():
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
        
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Warning: {missing_count} missing values remaining")
        return df
    
    def engineer_features(self, df):
        """Create additional features"""
        df = df.copy()
        
        # Age groups
        df['age_group'] = pd.cut(df['age'], bins=[0, 40, 55, 70, 100], labels=[0, 1, 2, 3])
        df['age_group'] = df['age_group'].astype(int)
        
        # High cholesterol flag
        df['high_chol'] = (df['chol'] > 240).astype(int)
        
        # High blood pressure flag
        df['high_bp'] = (df['trestbps'] > 140).astype(int)
        
        # Heart rate reserve (proxy for fitness)
        df['hr_reserve'] = 220 - df['age'] - df['thalach']
        
        return df
    
    def prepare_features(self, df, fit=True):
        """Scale numerical features and prepare final feature matrix"""
        df = df.copy()
        
        # Separate features and target
        if 'target' in df.columns:
            X = df.drop('target', axis=1)
            y = df['target']
        else:
            X = df
            y = None
        
        # Store feature names
        if self.feature_names is None:
            self.feature_names = X.columns.tolist()
        
        # Scale numerical features
        numerical_cols = [col for col in self.numerical_features + ['hr_reserve'] 
                         if col in X.columns]
        
        if fit:
            X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
        else:
            X[numerical_cols] = self.scaler.transform(X[numerical_cols])
        
        return X, y
    
    def preprocess(self, df, fit=True):
        """Complete preprocessing pipeline"""
        df = self.handle_missing_values(df)
        df = self.engineer_features(df)
        X, y = self.prepare_features(df, fit=fit)
        return X, y
    
    def save(self, filepath):
        """Save preprocessor"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features
        }, filepath)
        print(f"Preprocessor saved to {filepath}")
    
    def load(self, filepath):
        """Load preprocessor"""
        data = joblib.load(filepath)
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.numerical_features = data['numerical_features']
        self.categorical_features = data['categorical_features']
        print(f"Preprocessor loaded from {filepath}")


def prepare_train_test_data(data_path, test_size=0.2, random_state=42):
    """
    Load, preprocess, and split data into train/test sets
    """
    preprocessor = DataPreprocessor()
    
    # Load and preprocess
    df = preprocessor.load_data(data_path)
    X, y = preprocessor.preprocess(df, fit=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    print(f"Train distribution: {dict(y_train.value_counts())}")
    print(f"Test distribution: {dict(y_test.value_counts())}")
    
    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    # Test preprocessing
    data_path = "../data/heart_disease_raw.csv"
    X_train, X_test, y_train, y_test, preprocessor = prepare_train_test_data(data_path)
    
    # Save preprocessor
    preprocessor.save("../models/preprocessor.pkl")
