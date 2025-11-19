#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MLP 기반 다중 레이블 이상 탐지 시스템 - 예측/평가 스크립트

- train_mlp.py에서 학습 및 저장한 모델/스케일러/데이터 분할 결과를 불러와서
  테스트 세트 성능 평가 및 새 데이터 예측을 수행한다.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import pickle

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, multilabel_confusion_matrix

# =====================================
# 0. 설정 (학습 스크립트와 동일 디렉토리 구조 가정)
# =====================================
OUTPUT_DIR_MODELS = "models"
OUTPUT_DIR_RESULTS = "results/mlp"

MODEL_PATH = os.path.join(OUTPUT_DIR_MODELS, "mlp_classifier.pth")
SCALER_PATH = os.path.join(OUTPUT_DIR_MODELS, "scaler_mlp.pkl")
SPLIT_PATH = os.path.join(OUTPUT_DIR_MODELS, "data_split.npz")

# 레이블 매핑 (학습과 동일)
LABEL_MAPPING = {
    "fluctuating": [1, 0],
    "misalignment": [0, 1],
    "normal": [0, 0],
}

ANOMALY_LABELS = ["fluctuating", "misalignment"]  # 출력 벡터의 두 축


# =====================================
# 1. MLP 모델 정의 (train_mlp.py와 동일해야 함)
# =====================================
class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes=(64, 32), output_size=2):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))  # 출력은 [0, 1] 확률 벡터
        return x


# =====================================
# 2. 모델/스케일러/데이터 로드 유틸
# =====================================
def load_model_scaler_and_split():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")
    if not os.path.exists(SPLIT_PATH):
        raise FileNotFoundError(f"Data split not found: {SPLIT_PATH}")

    # 체크포인트 로드
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    input_size = checkpoint["input_size"]
    hidden_sizes = checkpoint.get("hidden_sizes", (64, 32))
    output_size = checkpoint.get("output_size", 2)

    # 모델 생성 및 가중치 로드
    model = MLPClassifier(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    # 스케일러 로드
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    # train/test 분할 데이터 로드
    split = np.load(SPLIT_PATH)
    X_train = split["X_train"]
    X_test = split["X_test"]
    y_train = split["y_train"]
    y_test = split["y_test"]

    return model, scaler, X_train, X_test, y_train, y_test, checkpoint


# =====================================
# 3. 테스트 세트 평가
# =====================================
def evaluate_on_test_set():
    print("\n" + "=" * 60)
    print("📂 Loading model / scaler / data split")
    print("=" * 60)

    model, scaler, X_train, X_test, y_train, y_test, checkpoint = load_model_scaler_and_split()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    print(f"✅ Using device: {device}")
    print(f"✅ X_train shape: {X_train.shape}")
    print(f"✅ X_test shape: {X_test.shape}")
    print(f"✅ y_train shape: {y_train.shape}")
    print(f"✅ y_test shape: {y_test.shape}")

    # Tensor 변환
    X_test_tensor = torch.FloatTensor(X_test).to(device)

    # 예측
    with torch.no_grad():
        y_pred_proba = model(X_test_tensor).cpu().numpy()   # (N, 2), 0~1
        y_pred = (y_pred_proba > 0.5).astype(int)           # threshold 0.5

    print("\n" + "=" * 60)
    print("📈 Classification Report (per-label, multi-label)")
    print("=" * 60)
    # 2개의 축: fluctuating / misalignment
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=ANOMALY_LABELS,
            digits=4,
            zero_division=0,
        )
    )

    # 전체 벡터 일치 정확도 (정상 [0,0] 포함)
    vector_acc = float(np.mean(np.all(y_test == y_pred, axis=1)))
    print(f"\n✅ Overall vector accuracy (including normal [0,0]): {vector_acc:.4f}")

    # 각 레이블(축) 별 confusion matrix 및 메트릭 저장
    print("\n" + "=" * 60)
    print("📊 Per-label metrics & confusion matrix")
    print("=" * 60)

    cm_all = multilabel_confusion_matrix(y_test, y_pred)

    os.makedirs(OUTPUT_DIR_RESULTS, exist_ok=True)

    for i, label_name in enumerate(ANOMALY_LABELS):
        cm = cm_all[i]
        tn, fp, fn, tp = cm.ravel()

        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        metrics = {
            "model_type": "MLP Classifier (Multi-Label, 2-dim)",
            "test_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "label": label_name,
            "data_split": {
                "train_samples": int(len(X_train)),
                "test_samples": int(len(X_test)),
            },
            "metrics": {
                "vector_accuracy_overall": vector_acc,
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
            },
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            },
        }

        metrics_path = os.path.join(OUTPUT_DIR_RESULTS, f"mlp_metrics_{label_name}.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        print(f"[{label_name}]")
        print(f"  TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        print(f"  accuracy = {accuracy:.4f}")
        print(f"  precision = {precision:.4f}")
        print(f"  recall = {recall:.4f}")
        print(f"  f1 = {f1:.4f}")
        print(f"  → saved: {metrics_path}")

    print("\n" + "=" * 60)
    print("✅ Test evaluation completed")
    print("=" * 60)

    # 예측 확률/벡터 자체도 npz로 덤프 (필요하면 anomaly-vector 분석에 사용)
    pred_dump_path = os.path.join(OUTPUT_DIR_RESULTS, "mlp_test_predictions.npz")
    np.savez(
        pred_dump_path,
        y_test=y_test,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba,
    )
    print(f"\n📁 Saved raw predictions: {pred_dump_path}")


# =====================================
# 4. 새 CSV에 대한 예측 유틸
# =====================================
def predict_from_csv(csv_path: str, model=None, scaler=None):
    """
    1118_features*_cleaned.csv 형태의 특징 벡터 CSV를 입력받아
    window별 이상도 벡터 / 예측 라벨을 반환하는 헬퍼 함수.

    - csv_path: window_id, start_time, end_time, feature1, feature2, ... 구조 가정
    - model, scaler를 인자로 넘기지 않으면 자동으로 로드함
    """
    if model is None or scaler is None:
        model, scaler, _, _, _, _, _ = load_model_scaler_and_split()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    df = pd.read_csv(csv_path)

    if df.shape[1] <= 3:
        raise ValueError("CSV에 feature 컬럼이 없습니다. (앞 3개 메타 컬럼 이후에 특징이 있어야 함)")

    meta_cols = df.iloc[:, :3]
    X_raw = df.iloc[:, 3:].values

    # 학습 때와 동일한 scaler 사용
    X_scaled = scaler.transform(X_raw)

    X_tensor = torch.FloatTensor(X_scaled).to(device)
    with torch.no_grad():
        y_proba = model(X_tensor).cpu().numpy()
        y_bin = (y_proba > 0.5).astype(int)

    # L2 norm (복합 이상도)
    vector_mag = np.linalg.norm(y_proba, axis=1)

    # 간단한 라벨 문자열로 매핑
    label_strs = []
    for vec in y_bin:
        if np.array_equal(vec, [0, 0]):
            label_strs.append("normal")
        elif np.array_equal(vec, [1, 0]):
            label_strs.append("fluctuating")
        elif np.array_equal(vec, [0, 1]):
            label_strs.append("misalignment")
        else:
            # [1,1] 등 복합 이상
            label_strs.append("composite")

    # 결과 DataFrame 구성
    result_df = meta_cols.copy()
    result_df["prob_fluctuating"] = y_proba[:, 0]
    result_df["prob_misalignment"] = y_proba[:, 1]
    result_df["anomaly_vector_norm"] = vector_mag
    result_df["pred_label"] = label_strs

    # 기본 저장 경로 (무조건 results/mlp 아래에 저장)
    os.makedirs(OUTPUT_DIR_RESULTS, exist_ok=True)
    
    filename = os.path.basename(csv_path)            # 예: 1118_features_cleaned.csv
    base, ext = os.path.splitext(filename)
    out_path = os.path.join(OUTPUT_DIR_RESULTS, base + "_predicted" + ext)
    
    result_df.to_csv(out_path, index=False)
    print(f"\n   → Saved with predictions: {out_path}")

    return result_df


# =====================================
# 5. main
# =====================================
if __name__ == "__main__":
    # 1) 먼저 테스트 세트 평가
    # evaluate_on_test_set()

    # 2) 필요하다면 아래 부분 수정해서 임의 CSV에 대한 예측 수행
    # 예시:
    # predict_from_csv("data/processed/1118_features_cleaned.csv")
    predict_from_csv("data/processed/1118_features_fluctuating_yellow_cleaned.csv")
    predict_from_csv("data/processed/1118_features_misalignment_yellow_cleaned.csv")