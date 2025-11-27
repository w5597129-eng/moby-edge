# src/mlp_predict.py
import os
import json
import numpy as np
import pandas as pd
import pickle
import onnxruntime as ort
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime

class MLP3LevelPredictor:
    """
    MLP 기반 3단계(Normal, Yellow, Red) 이상 탐지 시스템 예측기
    """
    def __init__(self, model_dir="models", result_dir="results/mlp_3level"):
        self.model_dir = model_dir
        self.result_dir = result_dir
        
        # 경로 설정
        self.onnx_path = os.path.join(model_dir, "mlp_classifier.onnx")
        self.scaler_path = os.path.join(model_dir, "scaler_mlp.pkl")
        self.split_path = os.path.join(model_dir, "data_split.npz")
        
        self.alert_levels = ['Normal', 'Yellow', 'Red']
        self.ort_session = None
        self.scaler = None
        
        # 초기화 시 모델 로드
        self._load_resources()

    def _load_resources(self):
        """모델 및 스케일러 로드"""
        if not os.path.exists(self.onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler not found: {self.scaler_path}")

        print(f"📦 Loading ONNX model from {self.onnx_path}...")
        self.ort_session = ort.InferenceSession(self.onnx_path)
        
        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        print("✅ Model and Scaler loaded successfully.")

    def evaluate_test_set(self):
        """저장된 Split 데이터를 사용하여 테스트셋 성능 평가"""
        if not os.path.exists(self.split_path):
            print(f"⚠️ Split data not found: {self.split_path}. Skipping evaluation.")
            return

        split = np.load(self.split_path)
        X_test = split["X_test"]
        y_test = split["y_test"]
        
        print(f"\n🔍 Evaluating on Test Set ({len(X_test)} samples)...")

        # 추론
        input_name = self.ort_session.get_inputs()[0].name
        X_test_float32 = X_test.astype(np.float32)
        y_pred_proba = self.ort_session.run(None, {input_name: X_test_float32})[0]

        # S1, S2 분리 및 argmax
        s1_pred = np.argmax(y_pred_proba[:, :3], axis=1)
        s2_pred = np.argmax(y_pred_proba[:, 3:], axis=1)
        s1_true = np.argmax(y_test[:, :3], axis=1)
        s2_true = np.argmax(y_test[:, 3:], axis=1)

        # 리포트 출력
        print("\n" + "=" * 60)
        print("📈 Classification Report - S1 (Fluctuation)")
        print("-" * 60)
        print(classification_report(s1_true, s1_pred, target_names=self.alert_levels, digits=4, zero_division=0))
        
        print("\n" + "=" * 60)
        print("📈 Classification Report - S2 (Unbalance)")
        print("-" * 60)
        print(classification_report(s2_true, s2_pred, target_names=self.alert_levels, digits=4, zero_division=0))

        # 메트릭 저장 로직 등은 필요시 추가 구현 가능

    def predict_csv(self, csv_path):
        """
        단일 CSV 파일에 대한 예측 수행 및 결과 저장
        """
        if not os.path.exists(csv_path):
            print(f"❌ File not found: {csv_path}")
            return None

        print(f"\n🔮 Processing: {csv_path}")
        df = pd.read_csv(csv_path)

        # 전처리 (메타데이터 분리)
        if df.shape[1] <= 3:
            raise ValueError("CSV has insufficient columns (features missing).")
            
        meta_cols = df.iloc[:, :3].copy()
        X_raw = df.iloc[:, 3:].values
        X_scaled = self.scaler.transform(X_raw)

        # 추론
        input_name = self.ort_session.get_inputs()[0].name
        X_float32 = X_scaled.astype(np.float32)
        y_proba = self.ort_session.run(None, {input_name: X_float32})[0]

        # 결과 정리
        result_df = meta_cols.copy()
        
        # S1 Probabilities
        result_df['s1_prob_normal'] = y_proba[:, 0]
        result_df['s1_prob_yellow'] = y_proba[:, 1]
        result_df['s1_prob_red']    = y_proba[:, 2]
        
        # S2 Probabilities
        result_df['s2_prob_normal'] = y_proba[:, 3]
        result_df['s2_prob_yellow'] = y_proba[:, 4]
        result_df['s2_prob_red']    = y_proba[:, 5]

        # 저장
        os.makedirs(self.result_dir, exist_ok=True)
        filename = os.path.basename(csv_path)
        base, ext = os.path.splitext(filename)
        out_path = os.path.join(self.result_dir, base + "_predicted_3level" + ext)
        
        result_df.to_csv(out_path, index=False)
        print(f"   → Saved prediction: {out_path}")
        
        self._print_summary(result_df)
        return out_path

    def _print_summary(self, df):
        """간단한 통계 요약 출력"""
        print(f"   [Summary] Total: {len(df)}")
        s1_mean = df[['s1_prob_normal', 's1_prob_yellow', 's1_prob_red']].mean()
        s2_mean = df[['s2_prob_normal', 's2_prob_yellow', 's2_prob_red']].mean()

        # s1_mean 수정 (위치 기반 접근으로 변경)
        print(f"   S1 Avg Probs: N={s1_mean.iloc[0]:.4f}, Y={s1_mean.iloc[1]:.4f}, R={s1_mean.iloc[2]:.4f}")
        
        # s2_mean 수정 (위치 기반 접근으로 변경)
        print(f"   S2 Avg Probs: N={s2_mean.iloc[0]:.4f}, Y={s2_mean.iloc[1]:.4f}, R={s2_mean.iloc[2]:.4f}")