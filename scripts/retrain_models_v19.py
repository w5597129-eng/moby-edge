#!/usr/bin/env python3
"""
V19 모델 재학습 스크립트
- 1205 데이터만 사용
- 13개 특징 (V19: Mean XYZ 추가)
- IsolationForest + StandardScaler + LightGBM
"""

import sys
import os
import glob
import json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from feature_extractor import process_multi_sensor_files

# V19 Feature Order (13개) - inference_interface.py와 동일해야 함
FEATURE_ORDER_V19 = [
    # Accel features (8)
    'accel_VectorRMS',
    'accel_PC1_PeakToPeak',
    'accel_PC1_DominantFreq',
    'accel_PC1_RMSF',
    'accel_PC1_VarianceRatio',
    'accel_Mean_X',
    'accel_Mean_Y',
    'accel_Mean_Z',
    # Gyro features (3)
    'gyro_STD_X',
    'gyro_STD_Y',
    'gyro_STD_Z',
    # IR Counter features (2)
    'ir_AvgCycleTime',
    'ir_CycleJitter',
]

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))

# 1205 데이터 클래스 정의
CLASSES = {
    0: {'name': 'Normal',    'pattern_accel': '1205_accel_gyro_normal.csv',    'pattern_ir': '1205_IRcounter_normal.csv'},
    1: {'name': 'S1_Yellow', 'pattern_accel': '1205_accel_gyro_s1_yellow.csv', 'pattern_ir': '1205_IRcounter_s1_yellow.csv'},
    2: {'name': 'S1_Red',    'pattern_accel': '1205_accel_gyro_s1_red.csv',    'pattern_ir': '1205_IRcounter_s1_red.csv'},
    3: {'name': 'S2_Yellow', 'pattern_accel': '1205_accel_gyro_s2_yellow*.csv', 'pattern_ir': '1205_IRcounter_s2_yellow*.csv'}, 
    4: {'name': 'S2_Red',    'pattern_accel': '1205_accel_gyro_s2_red.csv',    'pattern_ir': '1205_IRcounter_s2_red.csv'},
}


def load_all_data():
    """모든 1205 데이터를 로드하고 특징 추출"""
    all_X = []
    all_y = []

    for label, info in CLASSES.items():
        print(f"\n[PROCESSING] Class {label}: {info['name']}")
        
        accel_files = glob.glob(os.path.join(DATA_DIR, info['pattern_accel']))
        ir_files = glob.glob(os.path.join(DATA_DIR, info['pattern_ir']))
        
        accel_files.sort()
        ir_files.sort()
        
        if not accel_files or not ir_files:
            print(f"  [X] Missing files for {info['name']}")
            continue

        # 파일 매칭
        file_pairs = []
        for af in accel_files:
            base_name_accel = os.path.basename(af)
            tag = base_name_accel.replace('accel_gyro', '').replace('.csv', '').replace('1205_', '') 
            matching_irs = [f for f in ir_files if tag in os.path.basename(f)]
            
            if matching_irs:
                file_pairs.append({'accel_gyro': af, 'ir_counter': matching_irs[0]})
        
        if not file_pairs and len(accel_files) == 1 and len(ir_files) == 1:
            file_pairs.append({'accel_gyro': accel_files[0], 'ir_counter': ir_files[0]})

        print(f"  Found {len(file_pairs)} file pairs")

        for files in file_pairs:
            print(f"    Extracting from: {[os.path.basename(f) for f in files.values()]}")
            try:
                df = process_multi_sensor_files(
                    files, 
                    resample_rate='78.125ms',  # 12.8Hz
                    window_size=10.0, 
                    window_overlap=5.0
                )
                
                if df.empty:
                    print("    [!] No features extracted.")
                    continue
                
                # V19 특징 순서대로 추출
                missing_cols = [c for c in FEATURE_ORDER_V19 if c not in df.columns]
                if missing_cols:
                    print(f"    [!] Missing columns: {missing_cols}")
                    continue
                    
                X_part = df[FEATURE_ORDER_V19].values
                y_part = np.full(len(X_part), label)
                
                all_X.append(X_part)
                all_y.append(y_part)
                print(f"    -> Extracted {len(X_part)} windows with {X_part.shape[1]} features.")
                
            except Exception as e:
                print(f"    [!] Error: {e}")
                import traceback
                traceback.print_exc()

    if not all_X:
        print("No training data collected!")
        return None, None

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    
    return X, y


def train_models(X, y):
    """StandardScaler, IsolationForest, LightGBM 학습"""
    
    print("\n" + "=" * 60)
    print(f"TOTAL DATASET: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class Distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    print("=" * 60)
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    
    # 1. StandardScaler 학습
    print("\n[1/3] Training StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 2. IsolationForest 학습 (Normal 데이터만)
    print("[2/3] Training IsolationForest (on Normal class only)...")
    normal_mask = y_train == 0
    X_normal_scaled = X_train_scaled[normal_mask]
    
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.1,
        random_state=42,
        n_jobs=-1
    )
    iso_forest.fit(X_normal_scaled)
    
    # IsolationForest 평가
    y_pred_if = iso_forest.predict(X_test_scaled)
    y_pred_if_binary = (y_pred_if == -1).astype(int)  # -1 = anomaly
    y_test_binary = (y_test != 0).astype(int)  # 0 = normal, else = anomaly
    
    print("\n[IsolationForest Evaluation]")
    print(f"  Normal samples correctly identified: {np.sum((y_pred_if == 1) & (y_test == 0))}/{np.sum(y_test == 0)}")
    print(f"  Anomaly samples detected: {np.sum((y_pred_if == -1) & (y_test != 0))}/{np.sum(y_test != 0)}")
    
    # 3. LightGBM 학습
    print("\n[3/3] Training LightGBM Classifier...")
    try:
        import lightgbm as lgb
        
        clf = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=len(CLASSES),
            n_estimators=100,
            random_state=42,
            verbose=-1
        )
        clf.fit(X_train_scaled, y_train)
        
        y_pred = clf.predict(X_test_scaled)
        print("\n[LightGBM Evaluation]")
        print(classification_report(
            y_test, y_pred, 
            target_names=[CLASSES[i]['name'] for i in sorted(CLASSES.keys())]
        ))
        
        # Feature Importance
        print("\nFeature Importance:")
        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        for i in indices[:5]:
            print(f"  {FEATURE_ORDER_V19[i]:<25}: {importances[i]}")
        
        has_lgbm = True
    except ImportError:
        print("  [!] LightGBM not installed, skipping...")
        clf = None
        has_lgbm = False
    
    return scaler, iso_forest, clf, X_train, X_test, y_train, y_test


def save_models(scaler, iso_forest, clf, X_train, X_test, y_train, y_test):
    """모델 저장"""
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Scaler 저장
    scaler_path = os.path.join(MODEL_DIR, 'scaler_if.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"\n[SAVED] Scaler: {scaler_path}")
    
    # IsolationForest 저장
    if_path = os.path.join(MODEL_DIR, 'isolation_forest.joblib')
    joblib.dump(iso_forest, if_path)
    print(f"[SAVED] IsolationForest: {if_path}")
    
    # LightGBM 저장
    if clf is not None:
        lgbm_path = os.path.join(MODEL_DIR, 'lgbm_classifier.joblib')
        joblib.dump(clf, lgbm_path)
        print(f"[SAVED] LightGBM: {lgbm_path}")
    
    # 데이터 분할 저장
    split_path = os.path.join(MODEL_DIR, 'data_split.npz')
    np.savez(split_path, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    print(f"[SAVED] Data split: {split_path}")
    
    # Training Summary 저장
    summary = {
        "model_type": "IsolationForest + LightGBM (V19)",
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "feature_version": "V19",
        "feature_count": len(FEATURE_ORDER_V19),
        "feature_order": FEATURE_ORDER_V19,
        "data_sources": "1205 data only",
        "class_mapping": {str(k): v['name'] for k, v in CLASSES.items()},
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "scaler_path": "scaler_if.joblib",
        "isolation_forest_path": "isolation_forest.joblib",
        "lgbm_path": "lgbm_classifier.joblib" if clf else None,
    }
    
    summary_path = os.path.join(MODEL_DIR, 'training_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    print(f"[SAVED] Training summary: {summary_path}")


def main():
    print("=" * 60)
    print("V19 Model Retraining (1205 Data, 13 Features)")
    print("=" * 60)
    
    # 데이터 로드
    X, y = load_all_data()
    
    if X is None:
        print("\n[ERROR] Failed to load data!")
        return 1
    
    # 모델 학습
    scaler, iso_forest, clf, X_train, X_test, y_train, y_test = train_models(X, y)
    
    # 저장
    save_models(scaler, iso_forest, clf, X_train, X_test, y_train, y_test)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print(f"Models saved to: {MODEL_DIR}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
