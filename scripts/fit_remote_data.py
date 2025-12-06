
import os
import sys
import glob
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Setup Path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

try:
    from src.feature_extractor import extract_features, FEATURE_CONFIG_V19
except ImportError:
    print("Error: Could not import feature_extractor. Run this script from project root.")
    sys.exit(1)

# Configuration
RAW_DATA_DIR = "data/training_raw"
MODEL_DIR = "models"
WINDOW_SIZE_SEC = 1.0 # 1 second window
OVERLAP_SEC = 0.5     # 50% overlap

# Label Mapping
LABEL_MAP = {
    "normal": 0,
    "yellow": 1,
    "red": 2
}

def load_data_and_extract_features():
    print(f"Loading data from {RAW_DATA_DIR}...")
    X = []
    y = []
    
    files = glob.glob(os.path.join(RAW_DATA_DIR, "*.csv"))
    if not files:
        print("No CSV files found. Please run 'scripts/remote_data_collector.py' first.")
        return np.array([]), np.array([])
        
    for f in files:
        filename = os.path.basename(f)
        # Parse label from filename (start with label_)
        label_str = filename.split('_')[0].lower()
        
        # Extended mapping support
        if "normal" in filename.lower(): label_str = "normal"
        elif "yellow" in filename.lower(): label_str = "yellow"
        elif "red" in filename.lower(): label_str = "red"
        
        if label_str not in LABEL_MAP:
            print(f"Skipping {filename}: Unknown label (expected normal/yellow/red in filename)")
            continue
            
        target = LABEL_MAP[label_str]
        print(f"Processing {filename} as Class {target} ({label_str})...")
        
        df = pd.read_csv(f)
        
        # Calculate Sampling Data
        timestamps = df['timestamp_ns'].values
        if len(timestamps) < 2: continue
        
        # Convert timestamp_ns to float seconds for duration calc
        duration = (timestamps[-1] - timestamps[0]) / 1e9
        avg_sr = len(timestamps) / duration if duration > 0 else 100.0
        
        # Sliding Window
        window_samples = int(WINDOW_SIZE_SEC * avg_sr)
        step_samples = int(window_samples * (1 - OVERLAP_SEC))
        
        if window_samples > len(df):
            print(f"  Warning: File too short ({duration:.2f}s) for window {WINDOW_SIZE_SEC}s")
            continue

        # Prepare Data Dict
        acc_data = df[['accel_x', 'accel_y', 'accel_z']].values
        gyro_data = df[['gyro_x', 'gyro_y', 'gyro_z']].values
        
        n_windows = 0
        for start in range(0, len(df) - window_samples + 1, step_samples):
            end = start + window_samples
            
            # Slice
            w_acc = acc_data[start:end]
            w_gyro = gyro_data[start:end]
            
            data_dict = {
                'accel': w_acc,
                'gyro': w_gyro
                # IR not included in this simplified remote collector yet
            }
            
            try:
                # Extract V19 Features
                feats = extract_features(data_dict, avg_sr)
                
                # Convert to vector (Ordering matters!)
                # Must match inference_worker.py ordered_keys
                ordered_keys = [
                    'accel_VectorRMS', 'accel_PC1_PeakToPeak', 'accel_PC1_DominantFreq', 
                    'accel_PC1_BandEnergy_Low', 'accel_PC1_BandEnergy_High', 
                    'accel_PC1_SpectralEntropy', 'accel_PC1_RMSF', 'accel_PC1_VarianceRatio',
                    'gyro_STD_X', 'gyro_STD_Y', 'gyro_STD_Z',
                    'ir_AvgCycleTime' # Will be 0 if missing
                ]
                
                vec = [float(feats.get(k, 0.0)) for k in ordered_keys]
                X.append(vec)
                y.append(target)
                n_windows += 1
            except Exception as e:
                pass
                
        print(f"  -> Extracted {n_windows} windows")

    return np.array(X), np.array(y)

def train():
    print("=== Training Remote Data Model (RandomForest) ===")
    
    X, y = load_data_and_extract_features()
    
    if len(X) == 0:
        print("No Valid Data Found. Waiting for collection...")
        return
        
    print(f"\nTotal Samples: {len(X)}")
    print(f"Class Distribution: {np.bincount(y)}")
    
    if len(np.unique(y)) < 2:
         print("Error: Need at least 2 classes to train.")
         return

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Train
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_train_s, y_train)
    
    # Evaluate
    print("\nEvaluation:")
    y_pred = clf.predict(X_test_s)
    # Ensure all classes are present in classification report logic handling
    try:
        print(classification_report(y_test, y_pred))
    except Exception:
        print("Classification report error (possibly single class in test set)")
    
    # Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, os.path.join(MODEL_DIR, 'mlp_classifier.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler_mlp.pkl'))
    
    print(f"\n[SUCCESS] Model saved to {MODEL_DIR}/mlp_classifier.pkl")

if __name__ == "__main__":
    train()
