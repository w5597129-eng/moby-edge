# MOBY Edge Sensor & Inference System

라즈베리파이 기반 IoT 엣지 노드에서 센서 데이터를 수집하고, 실시간 이상 탐지 및 예지보전(RUL)을 수행하는 시스템입니다.

**최종 업데이트:** 2025-12-02

---

## 📋 주요 기능

| 기능 | 설명 |
|------|------|
| **센서 수집** | DHT11, 진동, 음향, MPU6050, BMP180 |
| **실시간 추론** | Isolation Forest, MLP Classifier (ONNX) |
| **예지보전** | IR 센서 기반 RUL(잔존 수명) 예측 |
| **통신** | MQTT 기반 pub/sub 아키텍처 |

---

## 🏗️ 시스템 구조

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  motor_PdM.py   │    │sensor_final.py  │    │inference_worker │
│  (모터 + RUL)   │    │ (센서 수집)      │    │  (추론 엔진)     │
└────────┬────────┘    └────────┬────────┘    └────────┬────────┘
         │                      │                      │
         └──────────────────────┼──────────────────────┘
                                │
                         ┌──────▼──────┐
                         │ MQTT Broker │
                         │192.168.80.143│
                         └─────────────┘
```

---

## 📁 디렉토리 구조

```
python/
├── src/                        # 핵심 소스 코드
│   ├── sensor_final.py         # 센서 수집 + 윈도우 생성
│   ├── inference_worker.py     # 추론 엔진
│   ├── inference_interface.py  # 메시지 스키마
│   ├── feature_extractor.py    # V17 특징 추출 (15개)
│   ├── predict_mlp.py          # MLP 예측기
│   └── predict_if.py           # IF 예측기
│
├── models/                     # 학습된 모델
│   ├── mlp_classifier.onnx     # MLP 모델 (ONNX)
│   ├── isolation_forest.joblib # IF 모델
│   ├── scaler_mlp.pkl          # MLP 스케일러
│   └── scaler_if.joblib        # IF 스케일러
│
├── motor_PdM.py                # 모터 + RUL 예측 (v1)
├── motor_PdM_v2.py             # 모터 + RUL 예측 (v2, 개선)
│
├── docs/                       # 문서
│   ├── inference.md            # 추론 시스템 문서
│   ├── motor_PdM_v2_changes.md # v2 변경 사항
│   └── 실행 구조.md            # 시스템 구조 문서
│
├── tests/                      # 테스트
│   └── test_inference_flow.py
│
├── scripts/                    # 유틸리티 스크립트
├── config/                     # 설정 파일
└── dummies/                    # 개발/테스트용 더미 센서
```

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성 (권장)
python3 -m venv .venv
source .venv/bin/activate  # Linux
# .venv\Scripts\activate   # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. 개별 실행

```bash
# 터미널 1: 모터 제어 + RUL 예측
sudo python motor_PdM.py

# 터미널 2: 센서 수집
sudo python src/sensor_final.py

# 터미널 3: 추론 워커
python src/inference_worker.py
```

### 3. 통합 실행

**Windows:**
```powershell
.\run_all.bat
```

**Linux/Raspberry Pi:**
```bash
bash run_all.sh
```

**Python (크로스 플랫폼):**
```bash
python run_all.py
```

---

## ⚙️ 주요 설정

| 항목 | 값 | 설명 |
|------|-----|------|
| `MQTT_BROKER` | 192.168.80.143 | MQTT 브로커 주소 |
| `FREQ_IMU` | 12.8 Hz | IMU 샘플링 주파수 |
| `WINDOW_SIZE` | 10.0초 | 추론 윈도우 크기 |
| `WINDOW_OVERLAP` | 5.0초 | 윈도우 겹침 |
| `EXPECTED_FEATURE_COUNT` | 15 | V17 특징 수 |

---

## 📡 MQTT 토픽

### 센서 데이터
| 토픽 | 주파수 | 내용 |
|------|--------|------|
| `factory/sensor/dht11` | 1 Hz | 습도 |
| `factory/sensor/vibration` | 12.8 Hz | 진동 |
| `factory/sensor/sound` | 12.8 Hz | 음향 |
| `factory/sensor/accel_gyro` | 12.8 Hz | 가속도/자이로 |
| `factory/sensor/pressure` | 1 Hz | 기압/온도 |

### 추론
| 토픽 | 내용 |
|------|------|
| `factory/inference/windows/accel_gyro` | 윈도우 메시지 |
| `factory/inference/results/accel_gyro/*` | 추론 결과 |

### 모터/RUL
| 토픽 | 내용 |
|------|------|
| `factory/conveyor/ir` | 회전 주기, RUL 예측 |

---

## 🔧 특징 추출 (V17)

총 **15개 특징** 추출:

| 센서 | 개수 | 특징 |
|------|------|------|
| Accel | 9 | VectorRMS, PC1_PeakToPeak, VectorCrestFactor, PC1_DominantFreq, PC1_RMSF, PC1_VarianceRatio, PC1_Direction_X/Y/Z |
| Gyro | 4 | VectorRMS, STD_X, STD_Y, STD_Z |
| Env | 2 | pressure_Mean, temperature_Mean |

---

## 🧪 테스트

```bash
# 전체 테스트
python -m pytest tests/ -v

# 특징 순서 검증 테스트
python -m pytest tests/test_inference_flow.py::test_feature_order_matches_training_csv -v
```

---

## 📖 문서

| 문서 | 설명 |
|------|------|
| [docs/inference.md](docs/inference.md) | 추론 시스템 상세 |
| [docs/실행 구조.md](docs/실행%20구조.md) | 시스템 아키텍처 |
| [docs/motor_PdM_v2_changes.md](docs/motor_PdM_v2_changes.md) | RUL v2 변경사항 |

---

## 🐛 트러블슈팅

### GPIO 권한 오류
```bash
sudo python src/sensor_final.py
# 또는 사용자를 gpio 그룹에 추가
sudo usermod -aG gpio $USER
```

### 모듈 import 오류
```bash
# 프로젝트 루트에서 실행
cd /home/wise/python
python src/sensor_final.py
```

### 특징 디버깅
```bash
FEATURE_DEBUG=1 python src/inference_worker.py
```

---

## 📝 버전 히스토리

| 버전 | 날짜 | 변경 사항 |
|------|------|-----------|
| 2.0 | 2025-12-02 | 샘플링 레이트 12.8Hz 통일, V17 특징 순서 표준화, motor_PdM v2 추가 |
| 1.0 | 2025-11 | 초기 버전 |

---

## 📄 라이센스

WISE Team, Project MOBY
