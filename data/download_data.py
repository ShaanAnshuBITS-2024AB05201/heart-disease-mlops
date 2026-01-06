"""
Generate Heart Disease dataset matching UCI format
"""
import pandas as pd
import numpy as np
import os

def generate_heart_disease_data():
    """
    Generate a synthetic Heart Disease dataset based on UCI format
    This matches the structure and distributions of the original dataset
    """
    np.random.seed(42)
    n_samples = 303  # Same as original dataset
    
    # Generate features based on realistic distributions
    data = {
        'age': np.random.randint(29, 78, n_samples),
        'sex': np.random.choice([0, 1], n_samples, p=[0.32, 0.68]),  # 1=male, 0=female
        'cp': np.random.choice([0, 1, 2, 3], n_samples, p=[0.47, 0.17, 0.28, 0.08]),  # chest pain type
        'trestbps': np.random.randint(94, 200, n_samples),  # resting blood pressure
        'chol': np.random.randint(126, 564, n_samples),  # serum cholesterol
        'fbs': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),  # fasting blood sugar > 120 mg/dl
        'restecg': np.random.choice([0, 1, 2], n_samples, p=[0.48, 0.48, 0.04]),  # resting ECG results
        'thalach': np.random.randint(71, 202, n_samples),  # maximum heart rate achieved
        'exang': np.random.choice([0, 1], n_samples, p=[0.68, 0.32]),  # exercise induced angina
        'oldpeak': np.random.uniform(0, 6.2, n_samples).round(1),  # ST depression
        'slope': np.random.choice([0, 1, 2], n_samples, p=[0.21, 0.50, 0.29]),  # slope of peak exercise ST
        'ca': np.random.choice([0, 1, 2, 3], n_samples, p=[0.58, 0.22, 0.13, 0.07]),  # number of major vessels
        'thal': np.random.choice([0, 1, 2, 3], n_samples, p=[0.02, 0.06, 0.55, 0.37]),  # thalassemia
    }
    
    df = pd.DataFrame(data)
    
    # Generate target with correlation to features
    # Create risk score based on medical knowledge
    risk_score = (
        (df['age'] > 55).astype(int) * 0.3 +
        (df['sex'] == 1).astype(int) * 0.2 +
        (df['cp'] >= 2).astype(int) * 0.25 +
        (df['trestbps'] > 140).astype(int) * 0.15 +
        (df['chol'] > 240).astype(int) * 0.2 +
        (df['fbs'] == 1).astype(int) * 0.1 +
        (df['thalach'] < 120).astype(int) * 0.25 +
        (df['exang'] == 1).astype(int) * 0.3 +
        (df['oldpeak'] > 2).astype(int) * 0.2 +
        (df['ca'] > 0).astype(int) * 0.25 +
        np.random.uniform(-0.3, 0.3, n_samples)
    )
    
    # Convert to binary target (0 = no disease, 1 = disease present)
    df['target'] = (risk_score > 1.2).astype(int)
    
    # Introduce some missing values (like original dataset)
    missing_indices = np.random.choice(df.index, size=int(0.02 * len(df) * len(df.columns)), replace=False)
    missing_cols = np.random.choice(df.columns[:-1], size=len(missing_indices))
    for idx, col in zip(missing_indices, missing_cols):
        df.loc[idx, col] = np.nan
    
    # Save dataset
    output_path = os.path.join(os.path.dirname(__file__), 'heart_disease_raw.csv')
    df.to_csv(output_path, index=False)
    
    print(f"Dataset generated: {df.shape}")
    print(f"Features: {len(df.columns) - 1}")
    print(f"Target distribution: {df['target'].value_counts().to_dict()}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Saved to: {output_path}")
    
    return df

if __name__ == "__main__":
    generate_heart_disease_data()
