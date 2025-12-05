import sys
import os
import glob
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from feature_extractor import process_multi_sensor_files, FEATURE_CONFIG_V18

# Feature Columns (Order must match inference_interface.py)
FEATURE_ORDER = [
    'accel_VectorRMS',
    'accel_PC1_PeakToPeak',
    'accel_PC1_DominantFreq',
    'accel_PC1_RMSF',
    'accel_PC1_VarianceRatio',
    'gyro_STD_X',
    'gyro_STD_Y',
    'gyro_STD_Z',
    'ir_AvgCycleTime',
    'ir_CycleJitter',
]

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))

# Define Classes and File Patterns (1205 Data)
# Label Mapping:
# 0: Normal
# 1: S1_Yellow
# 2: S1_Red
# 3: S2_Yellow
# 4: S2_Red
CLASSES = {
    0: {'name': 'Normal',    'pattern_accel': '1205_accel_gyro_normal.csv',    'pattern_ir': '1205_IRcounter_normal.csv'},
    1: {'name': 'S1_Yellow', 'pattern_accel': '1205_accel_gyro_s1_yellow.csv', 'pattern_ir': '1205_IRcounter_s1_yellow.csv'},
    2: {'name': 'S1_Red',    'pattern_accel': '1205_accel_gyro_s1_red.csv',    'pattern_ir': '1205_IRcounter_s1_red.csv'},
    3: {'name': 'S2_Yellow', 'pattern_accel': '1205_accel_gyro_s2_yellow*.csv', 'pattern_ir': '1205_IRcounter_s2_yellow*.csv'}, # Handles _2
    4: {'name': 'S2_Red',    'pattern_accel': '1205_accel_gyro_s2_red.csv',    'pattern_ir': '1205_IRcounter_s2_red.csv'},
}

def main():
    print(f"Data Directory: {DATA_DIR}")
    
    all_X = []
    all_y = []

    for label, info in CLASSES.items():
        print(f"\nPROCESSING CLASS {label}: {info['name']}")
        
        # Find files
        accel_files = glob.glob(os.path.join(DATA_DIR, info['pattern_accel']))
        ir_files = glob.glob(os.path.join(DATA_DIR, info['pattern_ir']))
        
        accel_files.sort()
        ir_files.sort()
        
        if not accel_files or not ir_files:
            print(f"  [X] Missing files for {info['name']}")
            continue

        file_pairs = []
        for af in accel_files:
            # Try to find matching IR file
            # e.g. 1205_accel_gyro_s2_yellow_2.csv -> s2_yellow_2
            base_name_accel = os.path.basename(af)
            tag = base_name_accel.replace('accel_gyro', '').replace('.csv', '').replace('1205_', '') 
            # tag e.g. "normal", "s1_yellow", "s2_yellow_2"
            
            matching_irs = [f for f in ir_files if tag in os.path.basename(f)]
            
            if matching_irs:
                file_pairs.append({'accel_gyro': af, 'ir_counter': matching_irs[0]})
            else:
                pass
        
        if not file_pairs and len(accel_files)==1 and len(ir_files)==1:
             file_pairs.append({'accel_gyro': accel_files[0], 'ir_counter': ir_files[0]})

        print(f"  Found {len(file_pairs)} file pairs")

        for files in file_pairs:
            print(f"    Extracting from: {[os.path.basename(f) for f in files.values()]}")
            try:
                # Use training sampling rate (78.125ms = 12.8Hz) to match inference config
                df = process_multi_sensor_files(
                    files, 
                    resample_rate='78.125ms', 
                    window_size=10.0, 
                    window_overlap=5.0
                )
                
                if df.empty:
                    print("    [!] No features extracted.")
                    continue
                    
                # Select only relevant columns in order
                X_part = df[FEATURE_ORDER].values
                y_part = np.full(len(X_part), label)
                
                all_X.append(X_part)
                all_y.append(y_part)
                print(f"    -> Extracted {len(X_part)} windows.")
                
            except Exception as e:
                print(f"    [!] Error: {e}")

    if not all_X:
        print("No training data collected.")
        return

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    
    print("\n" + "="*50)
    print(f"TOTAL DATASET: {X.shape} samples")
    print(f"Class Distribution: {np.bincount(y)}")
    print("="*50 + "\n")

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train
    print("Training LightGBM...")
    clf = lgb.LGBMClassifier(objective='multiclass', num_class=len(CLASSES), n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    print("\nEvaluation Report:")
    print(classification_report(y_test, y_pred, target_names=[CLASSES[i]['name'] for i in sorted(CLASSES.keys())]))
    
    # Feature Importance
    print("\nFeature Importance:")
    importances = clf.feature_importances_
    # Normalize for readability
    importances_norm = importances / importances.sum()
    
    # Sort
    indices = np.argsort(importances)[::-1]
    for i in indices:
        print(f"{FEATURE_ORDER[i]:<25}: {importances[i]} ({importances_norm[i]*100:.1f}%)")

    # Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_DIR, 'lgbm_classifier.joblib')
    joblib.dump(clf, save_path)
    print(f"\nModel saved to: {save_path}")

if __name__ == "__main__":
    main()
