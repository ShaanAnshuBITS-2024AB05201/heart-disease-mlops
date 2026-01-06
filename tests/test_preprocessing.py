"""
Unit tests for Heart Disease Prediction Pipeline
Team Members: Shaan Anshu (2024AB05201)
"""
import pytest
import pandas as pd
import numpy as np
import joblib
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from preprocessing import DataPreprocessor, prepare_train_test_data


class TestDataPreprocessor:
    """Test data preprocessing functionality"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        return pd.DataFrame({
            'age': [63, 37, 41, 56, 57],
            'sex': [1, 1, 0, 1, 0],
            'cp': [3, 2, 1, 1, 0],
            'trestbps': [145, 130, 130, 120, 120],
            'chol': [233, 250, 204, 236, 354],
            'fbs': [1, 0, 0, 0, 0],
            'restecg': [0, 1, 0, 1, 1],
            'thalach': [150, 187, 172, 178, 163],
            'exang': [0, 0, 0, 0, 1],
            'oldpeak': [2.3, 3.5, 1.4, 0.8, 0.6],
            'slope': [0, 0, 2, 2, 2],
            'ca': [0, 0, 0, 0, 0],
            'thal': [1, 2, 2, 2, 2],
            'target': [1, 1, 1, 1, 0]
        })
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initializes correctly"""
        preprocessor = DataPreprocessor()
        assert preprocessor.scaler is not None
        assert len(preprocessor.numerical_features) > 0
        assert len(preprocessor.categorical_features) > 0
    
    def test_missing_value_handling(self, sample_data):
        """Test missing value imputation"""
        preprocessor = DataPreprocessor()
        
        # Add missing values
        data_with_missing = sample_data.copy()
        data_with_missing.loc[0, 'age'] = np.nan
        data_with_missing.loc[1, 'chol'] = np.nan
        
        # Handle missing values
        cleaned_data = preprocessor.handle_missing_values(data_with_missing)
        
        # Check no missing values remain
        assert cleaned_data.isnull().sum().sum() == 0
    
    def test_feature_engineering(self, sample_data):
        """Test feature engineering creates new features"""
        preprocessor = DataPreprocessor()
        engineered_data = preprocessor.engineer_features(sample_data)
        
        # Check new features are created
        assert 'age_group' in engineered_data.columns
        assert 'high_chol' in engineered_data.columns
        assert 'high_bp' in engineered_data.columns
        assert 'hr_reserve' in engineered_data.columns
    
    def test_preprocessing_pipeline(self, sample_data):
        """Test complete preprocessing pipeline"""
        preprocessor = DataPreprocessor()
        X, y = preprocessor.preprocess(sample_data, fit=True)
        
        # Check output shapes
        assert X.shape[0] == sample_data.shape[0]
        assert y is not None
        assert len(y) == sample_data.shape[0]
        
        # Check no missing values
        assert X.isnull().sum().sum() == 0
    
    def test_save_load_preprocessor(self, sample_data, tmp_path):
        """Test saving and loading preprocessor"""
        preprocessor = DataPreprocessor()
        preprocessor.preprocess(sample_data, fit=True)
        
        # Save
        save_path = tmp_path / "preprocessor.pkl"
        preprocessor.save(str(save_path))
        
        # Load
        new_preprocessor = DataPreprocessor()
        new_preprocessor.load(str(save_path))
        
        # Check attributes are loaded
        assert new_preprocessor.feature_names is not None
        assert new_preprocessor.scaler is not None


class TestModelPredictions:
    """Test model predictions"""
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input for prediction"""
        return pd.DataFrame([{
            'age': 63,
            'sex': 1,
            'cp': 3,
            'trestbps': 145,
            'chol': 233,
            'fbs': 1,
            'restecg': 0,
            'thalach': 150,
            'exang': 0,
            'oldpeak': 2.3,
            'slope': 0,
            'ca': 0,
            'thal': 1
        }])
    
    def test_prediction_output_shape(self, sample_input):
        """Test prediction output has correct shape"""
        # This test requires a trained model
        # For now, test the data preparation
        preprocessor = DataPreprocessor()
        X, _ = preprocessor.preprocess(sample_input, fit=False)
        
        assert X.shape[0] == 1
        assert X.shape[1] > 0


class TestDataValidation:
    """Test data validation"""
    
    def test_data_types(self):
        """Test data types are correct"""
        data = pd.DataFrame({
            'age': [63],
            'sex': [1],
            'cp': [3],
            'trestbps': [145],
            'chol': [233],
            'fbs': [1],
            'restecg': [0],
            'thalach': [150],
            'exang': [0],
            'oldpeak': [2.3],
            'slope': [0],
            'ca': [0],
            'thal': [1],
            'target': [1]
        })
        
        # Check numerical types
        assert pd.api.types.is_numeric_dtype(data['age'])
        assert pd.api.types.is_numeric_dtype(data['chol'])
        assert pd.api.types.is_numeric_dtype(data['oldpeak'])
    
    def test_data_ranges(self):
        """Test data values are within expected ranges"""
        data = pd.DataFrame({
            'age': [63],
            'sex': [1],
            'target': [1]
        })
        
        # Age should be positive
        assert (data['age'] > 0).all()
        
        # Sex should be 0 or 1
        assert data['sex'].isin([0, 1]).all()
        
        # Target should be 0 or 1
        assert data['target'].isin([0, 1]).all()


class TestTrainTestSplit:
    """Test train-test split functionality"""
    
    def test_split_sizes(self):
        """Test train-test split produces correct sizes"""
        # Create sample data file
        data = pd.DataFrame({
            'age': np.random.randint(30, 80, 100),
            'sex': np.random.choice([0, 1], 100),
            'cp': np.random.choice([0, 1, 2, 3], 100),
            'trestbps': np.random.randint(90, 200, 100),
            'chol': np.random.randint(120, 400, 100),
            'fbs': np.random.choice([0, 1], 100),
            'restecg': np.random.choice([0, 1, 2], 100),
            'thalach': np.random.randint(70, 200, 100),
            'exang': np.random.choice([0, 1], 100),
            'oldpeak': np.random.uniform(0, 5, 100),
            'slope': np.random.choice([0, 1, 2], 100),
            'ca': np.random.choice([0, 1, 2, 3], 100),
            'thal': np.random.choice([0, 1, 2, 3], 100),
            'target': np.random.choice([0, 1], 100)
        })
        
        # Save temporarily
        temp_path = '/tmp/test_data.csv'
        data.to_csv(temp_path, index=False)
        
        # Test split
        try:
            X_train, X_test, y_train, y_test, _ = prepare_train_test_data(
                temp_path, test_size=0.2
            )
            
            # Check sizes
            assert len(X_train) == 80
            assert len(X_test) == 20
            assert len(y_train) == 80
            assert len(y_test) == 20
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
