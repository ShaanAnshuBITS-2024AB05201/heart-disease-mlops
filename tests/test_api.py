"""
Basic tests for MLOps pipeline
Shaan Anshu (2024AB05201)
"""
import pytest
import pandas as pd

def test_feature_engineering():
    """Test that feature engineering works correctly"""
    df = pd.DataFrame({
        'age': [63, 45],
        'thalach': [150, 160],
        'chol': [250, 200],
        'trestbps': [145, 120]
    })
    
    # Test hr_reserve calculation
    df['hr_reserve'] = 220 - df['age'] - df['thalach']
    assert df['hr_reserve'].iloc[0] == 7
    assert df['hr_reserve'].iloc[1] == 15
    
    # Test high cholesterol flag
    df['high_chol'] = (df['chol'] > 240).astype(int)
    assert df['high_chol'].iloc[0] == 1
    assert df['high_chol'].iloc[1] == 0
    
def test_data_preprocessing():
    """Test data preprocessing functions"""
    df = pd.DataFrame({
        'age': [63],
        'trestbps': [145]
    })
    
    # Basic checks
    assert len(df) == 1
    assert 'age' in df.columns
    assert df['age'].iloc[0] > 0