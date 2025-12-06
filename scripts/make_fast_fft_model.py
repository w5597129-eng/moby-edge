
import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Add src to path
sys.path.append(os.getcwd())
try:
    from src.feature_extractor import extract_features
except ImportError:
    pass

# Direct definition of V19 keys (Must match inference_worker.py)
FEATURE_KEYS = [
    'accel_VectorRMS', 'accel_PC1_PeakToPeak', 'accel_PC1_DominantFreq', 
    'accel_PC1_BandEnergy_Low', 'accel_PC1_BandEnergy_High', 
    'accel_PC1_SpectralEntropy', 'accel_PC1_RMSF', 'accel_PC1_VarianceRatio',
    'gyro_STD_X', 'gyro_STD_Y', 'gyro_STD_Z',
    'ir_AvgCycleTime'
]

def train_model():
    print("=== Training Fast FFT Model (RandomForest) ===")
    
    # Simulation: 
    # Class 0 (Normal): Low RMS, Low Energy, DomFreq < 10Hz
    # Class 1 (Yellow): Med RMS, Med High Energy, DomFreq ~ 30Hz
    # Class 2 (Red): High RMS, High High Energy, DomFreq ~ 30-50Hz
    
    print("Generating Synthetic Pilot Data for Verification...")
    np.random.seed(42)
    n_samples = 300 # 100 per class
    
    X = []
    y_s1 = []
    
    for i in range(n_samples):
        # Base features
        vec = np.zeros(len(FEATURE_KEYS))
        
        # Determine class
        if i < 100: # Normal
            cls = 0
            # Low Vibration
            vec[0] = np.random.normal(9.8, 0.1) # RMS
            vec[2] = np.random.uniform(0, 5)   # Freq < 5Hz
            vec[4] = np.random.uniform(0, 100) # High Band Energy (Low)
        elif i < 200: # Yellow
            cls = 1
            # Medium Vibration at 30Hz
            vec[0] = np.random.normal(10.0, 0.3)
            vec[2] = np.random.normal(30, 2)    # Freq ~ 30Hz
            vec[4] = np.random.uniform(500, 1000) # High Band Energy (Med)
        else: # Red
            cls = 2
            # High Vibration
            vec[0] = np.random.normal(11.0, 0.5)
            vec[2] = np.random.normal(30, 5)    # Freq ~ 30Hz (Unbalance)
            vec[4] = np.random.uniform(2000, 5000) # High Band Energy (High)
            
        X.append(vec)
        y_s1.append(cls)
        
    X = np.array(X)
    y_s1 = np.array(y_s1)
    
    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y_s1, test_size=0.2, random_state=42)
    
    # 3. Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # 4. Train RandomForest
    # Using RF instead of MLP for faster training on small data and better robustness
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_train_s, y_train)
    
    # 5. Evaluate
    print("\nTraining Results (Synthetic Data):")
    y_pred = clf.predict(X_test_s)
    print(classification_report(y_test, y_pred))
    
    # 6. Save
    os.makedirs('models', exist_ok=True)
    joblib.dump(clf, 'models/mlp_classifier.pkl') # Save as pkl (compatible with inference_worker loader)
    joblib.dump(scaler, 'models/scaler_mlp.pkl')
    
    print("\n[SUCCESS] Model trained and saved to 'models/mlp_classifier.pkl'")
    print("NOTE: This model is trained on SYNTHETIC data to verify the pipeline.")
    print("      Run this script again after collecting REAL data to retrain.")

if __name__ == "__main__":
    train_model()
