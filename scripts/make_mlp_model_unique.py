
import os
import sys
import glob
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import joblib

# Setup
PROJECT_ROOT = os.getcwd()
sys.path.append(PROJECT_ROOT)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Config
HIDDEN_LAYER_SIZES = (64, 32)
MAX_ITER = 500
LEARNING_RATE_INIT = 0.001
ALPHA = 0.0001 # L2 regularization

def load_data():
    all_files = glob.glob(os.path.join(DATA_DIR, "*_features.csv"))
    dfs = []
    
    print(f"Loading data from {len(all_files)} files...")
    
    for fpath in all_files:
        fname = os.path.basename(fpath)
        df = pd.read_csv(fpath)
        
        # Parse labels from filename
        # Labels: Normal, S1_Yellow, S1_Red, S2_Yellow, S2_Red
        # Target format: Multi-label output for compatibility [S1_N, S1_Y, S1_R, S2_N, S2_Y, S2_R]
        
        label_vec = [1, 0, 0, 1, 0, 0] # Default Normal
        
        if "s1_yellow" in fname.lower():
            label_vec = [0, 1, 0, 1, 0, 0]
        elif "s1_red" in fname.lower():
            label_vec = [0, 0, 1, 1, 0, 0]
        elif "s2_yellow" in fname.lower():
            label_vec = [1, 0, 0, 0, 1, 0]
        elif "s2_red" in fname.lower():
            label_vec = [1, 0, 0, 0, 0, 1]
            
        # Add label columns
        for i, val in enumerate(label_vec):
            df[f'target_{i}'] = val
            
        dfs.append(df)
        
    if not dfs:
        raise ValueError("No feature files found in data/processed!")
        
    full_df = pd.concat(dfs, ignore_index=True)
    return full_df

def train():
    # 1. Load Data
    df = load_data()
    
    feature_cols = [c for c in df.columns if c not in ['window_id', 'start_time', 'end_time'] and not c.startswith('target_')]
    target_cols = [c for c in df.columns if c.startswith('target_')]
    
    X = df[feature_cols].values
    y = df[target_cols].values # (N, 6)
    
    print(f"Data Shape: X={X.shape}, y={y.shape}")
    
    # 2. Split and Scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save Scaler
    scaler_path = os.path.join(MODELS_DIR, "scaler_mlp.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")
        
    # 3. Setup Model (Scikit-Learn MLP)
    # MLPClassifier doesn't support multi-label indicator matrix easily with softmax output structure intended for 2 groups.
    # However, inference_worker expects predict_proba to return shape (N, 6) or list of arrays.
    # If we train MultiOutputClassifier(MLPClassifier), we get a list of arrays.
    # If we train a single MLPClassifier on 'multilabel-indicator', output activation is logistic (sigmoid).
    # This matches our target vectors (0/1).
    
    clf = MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYER_SIZES,
        activation='relu',
        solver='adam',
        alpha=ALPHA,
        batch_size=32,
        learning_rate_init=LEARNING_RATE_INIT,
        max_iter=MAX_ITER,
        random_state=42,
        early_stopping=True
    )
    
    print("Starting training (Sklearn MLP)...")
    clf.fit(X_train_scaled, y_train)
    
    print("Training Score:", clf.score(X_train_scaled, y_train))
    print("Test Score:", clf.score(X_test_scaled, y_test))
    
    # 4. Evaluate
    y_pred = clf.predict(X_test_scaled)
    y_pred_proba = clf.predict_proba(X_test_scaled) # Shape (N, 6)
    
    # Check predictions
    # S1 (idx 0-2), S2 (idx 3-5)
    s1_true = np.argmax(y_test[:, :3], axis=1)
    s1_pred = np.argmax(y_pred_proba[:, :3], axis=1)
    s2_true = np.argmax(y_test[:, 3:], axis=1)
    s2_pred = np.argmax(y_pred_proba[:, 3:], axis=1)
    
    print("\n--- S1 Classification Report ---")
    print(classification_report(s1_true, s1_pred, target_names=['Normal', 'Yellow', 'Red'], zero_division=0))
    print("\n--- S2 Classification Report ---")
    print(classification_report(s2_true, s2_pred, target_names=['Normal', 'Yellow', 'Red'], zero_division=0))
    
    # 5. Export
    model_path = os.path.join(MODELS_DIR, "mlp_classifier.pkl")
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train()
