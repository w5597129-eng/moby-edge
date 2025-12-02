# Inference System Documentation

## 📋 개요

MOBY Edge 시스템의 실시간 이상 탐지 추론 파이프라인 문서입니다.

**최종 업데이트:** 2025-12-02

---

## 🏗️ 아키텍처

```
┌─────────────────────┐     MQTT      ┌──────────────────────┐
│  sensor_final.py    │ ────────────→ │  inference_worker.py │
│  (센서 수집)         │   Windows     │  (모델 추론)          │
│                     │               │                      │
│  - MPU6050 (12.8Hz) │               │  - Isolation Forest  │
│  - BMP180           │               │  - MLP Classifier    │
│  - Vibration        │               │                      │
└─────────────────────┘               └──────────────────────┘
         ▲                                      │
         │              MQTT Results            │
         └──────────────────────────────────────┘
```

---

## 📡 MQTT 토픽 구조

### 센서 데이터 토픽 (Raw)
| 센서 | 토픽 | 주파수 |
|------|------|--------|
| DHT11 (습도) | `factory/sensor/dht11` | 1 Hz |
| 진동 (SEN0209) | `factory/sensor/vibration` | 12.8 Hz |
| 음향 (ADS1115) | `factory/sensor/sound` | 12.8 Hz |
| IMU (MPU6050) | `factory/sensor/accel_gyro` | 12.8 Hz |
| 기압 (BMP180) | `factory/sensor/pressure` | 1 Hz |

### 추론 관련 토픽
| 용도 | 토픽 |
|------|------|
| 윈도우 메시지 | `factory/inference/windows/accel_gyro` |
| 추론 결과 (통합) | `factory/inference/results/accel_gyro` |
| IF 모델 결과 | `factory/inference/results/accel_gyro/isolation_forest` |
| MLP 모델 결과 | `factory/inference/results/accel_gyro/mlp_classifier` |

---

## 📦 메시지 형식

### WindowMessage (센서 → 워커)

```json
{
  "kind": "sensor_window",
  "sensor_type": "accel_gyro",
  "sampling_rate_hz": 12.8,
  "window_fields": {
    "fields_accel_x": [0.12, 0.13, ...],
    "fields_accel_y": [-0.01, -0.02, ...],
    "fields_accel_z": [0.98, 0.97, ...],
    "fields_gyro_x": [0.01, 0.02, ...],
    "fields_gyro_y": [-0.01, -0.02, ...],
    "fields_gyro_z": [0.00, 0.01, ...],
    "fields_pressure_hpa": [1013.2, 1013.3, ...],
    "fields_temperature_c": [25.1, 25.2, ...]
  },
  "timestamp_ns": 1733100000000000000,
  "context_payload": { ... }
}
```

### InferenceResultMessage (워커 → 센서)

```json
{
  "kind": "inference_result",
  "sensor_type": "accel_gyro",
  "score": 0.7234,
  "model_name": "mlp_classifier",
  "timestamp_ns": 1733100000000000000,
  "context_payload": {
    "fields": {
      "mlp_score": 0.7234,
      "mlp_s1_prob_normal": 0.85,
      "mlp_s1_prob_yellow": 0.10,
      "mlp_s1_prob_red": 0.05,
      "mlp_s2_prob_normal": 0.90,
      "mlp_s2_prob_yellow": 0.07,
      "mlp_s2_prob_red": 0.03
    }
  }
}
```

---

## 🔧 특징 추출 (V17)

### 특징 목록 (15개)

| 센서 | 특징명 | 설명 |
|------|--------|------|
| **Accel (9)** | `accel_VectorRMS` | 3축 벡터 RMS (총 진동 에너지) |
| | `accel_PC1_PeakToPeak` | PC1 최대 진폭 |
| | `accel_VectorCrestFactor` | 벡터 충격도 |
| | `accel_PC1_DominantFreq` | PC1 주파수 |
| | `accel_PC1_RMSF` | PC1 고주파 이동 |
| | `accel_PC1_VarianceRatio` | PC1 설명력 |
| | `accel_PC1_Direction_X/Y/Z` | PC1 방향 벡터 |
| **Gyro (4)** | `gyro_VectorRMS` | 회전 불안정성 총량 |
| | `gyro_STD_X/Y/Z` | 축별 회전 속도 변동성 |
| **Env (2)** | `pressure_Mean` | 평균 기압 |
| | `temperature_Mean` | 평균 온도 |

### 특징 순서 (FEATURE_ORDER_V17)

```python
FEATURE_ORDER_V17 = [
    'accel_VectorRMS', 'accel_PC1_PeakToPeak', 'accel_VectorCrestFactor',
    'accel_PC1_DominantFreq', 'accel_PC1_RMSF', 'accel_PC1_VarianceRatio',
    'accel_PC1_Direction_X', 'accel_PC1_Direction_Y', 'accel_PC1_Direction_Z',
    'gyro_VectorRMS', 'gyro_STD_X', 'gyro_STD_Y', 'gyro_STD_Z',
    'pressure_Mean', 'temperature_Mean'
]
```

> ⚠️ **중요:** 학습 CSV와 실시간 추론의 특징 순서가 반드시 일치해야 합니다.

---

## 🤖 모델 구성

### Isolation Forest
- **파일:** `models/isolation_forest.joblib`
- **스케일러:** `models/scaler_if.joblib`
- **출력:** `iforest_score` (이상 점수, 낮을수록 이상)

### MLP Classifier (3-Level Alert)
- **파일:** `models/mlp_classifier.onnx`
- **스케일러:** `models/scaler_mlp.pkl`
- **출력:** 6개 확률값
  - S1 (Fluctuation): `prob_normal`, `prob_yellow`, `prob_red`
  - S2 (Unbalance): `prob_normal`, `prob_yellow`, `prob_red`

---

## ⚙️ 설정 파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `WINDOW_SIZE` | 10.0초 | 추론 윈도우 크기 |
| `WINDOW_OVERLAP` | 5.0초 | 윈도우 겹침 |
| `FREQ_IMU` | 12.8 Hz | IMU 샘플링 주파수 |
| `EXPECTED_FEATURE_COUNT` | 15 | V17 특징 수 |

---

## 🚀 실행 방법

### 센서 퍼블리셔 (Raspberry Pi)
```bash
cd /home/wise/python
sudo python src/sensor_final.py
```

### 추론 워커 (별도 터미널)
```bash
cd /home/wise/python
python src/inference_worker.py
```

### 결과 모니터링
```bash
mosquitto_sub -h 192.168.80.143 -t "factory/inference/results/#" -v
```

---

## 🐛 트러블슈팅

### 특징 수 불일치 오류
```
[FEATURE_VECTOR] ⚠️ V17 feature count mismatch: got 13, expected 15
```
→ pressure/temperature 데이터가 누락됨. BMP180 센서 연결 확인.

### 모델 로드 실패
```
Failed to load ONNX model: ...
```
→ `models/mlp_classifier.onnx` 파일 존재 및 권한 확인.

### 특징 순서 디버깅
```bash
FEATURE_DEBUG=1 python src/inference_worker.py
```

---

## 📊 성능 고려사항

1. **샘플링 레이트 일치:** 학습 데이터(12.8Hz)와 실시간 수집(12.8Hz) 일치 필수
2. **윈도우 크기:** 10초 윈도우 = 약 128개 샘플
3. **추론 지연:** 윈도우 채움 시간(5~10초) + 모델 추론 시간(<100ms)

