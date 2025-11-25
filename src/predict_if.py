# src/if_predict.py
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

class IsoForestPredictor:
    """
    Isolation Forest 기반 비지도 이상 탐지 예측기
    """
    def __init__(self, model_path='models/isolation_forest.joblib',
                 scaler_path='models/scaler_if.joblib', result_dir='results/isolation_forest'):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.result_dir = result_dir
        
        self.model = None
        self.scaler = None
        
        self._load_model()

    def _load_model(self):
        """모델 및 스케일러 로드"""
        if not os.path.exists(self.model_path) or not os.path.exists(self.scaler_path):
            raise FileNotFoundError("Model or Scaler file missing.")
            
        print(f"📦 Loading Isolation Forest model...")
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
            
        with open(self.scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print("✅ Model loaded successfully.")

    def predict(self, input_csv, output_filename='predictions.csv'):
        """
        CSV 파일에 대해 이상 탐지 수행
        """
        if not os.path.exists(input_csv):
            print(f"❌ Input file not found: {input_csv}")
            return

        print(f"\n📂 Loading Data: {input_csv}")
        df = pd.read_csv(input_csv)
        
        # 메타데이터 제외하고 특징 추출 (앞 3열이 메타데이터라고 가정)
        # 실제 데이터 구조에 따라 이 부분은 조정 필요할 수 있음
        # 원본 코드 로직: "metadata_cols" 제외한 나머지
        metadata_cols = ['window_id', 'start_time', 'end_time']
        feature_cols = [col for col in df.columns if col not in metadata_cols]
        
        X = df[feature_cols].values
        X_scaled = self.scaler.transform(X)
        
        print("🔍 Making Predictions...")
        # 이상 점수 및 예측
        anomaly_scores = self.model.score_samples(X_scaled)
        predictions = self.model.predict(X_scaled) # 1: 정상, -1: 이상
        
        # 결과 DataFrame 생성
        df_results = df.copy()
        df_results['anomaly_score'] = anomaly_scores
        df_results['prediction'] = predictions
        df_results['prediction_label'] = ['Normal' if p == 1 else 'Anomaly' for p in predictions]
        df_results['prediction_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 통계 출력
        n_total = len(predictions)
        n_anomaly = np.sum(predictions == -1)
        print(f"   Total: {n_total}, Anomalies: {n_anomaly} ({n_anomaly/n_total*100:.1f}%)")
        
        # 저장
        os.makedirs(self.result_dir, exist_ok=True)
        out_path = os.path.join(self.result_dir, output_filename)
        df_results.to_csv(out_path, index=False)
        print(f"💾 Results saved to: {out_path}")
        
        return df_results