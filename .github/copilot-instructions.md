# AI 코딩 에이전트를 위한 가이드

## 프로젝트 개요
MOBY Edge Sensor Publisher: IoT 센서 데이터를 라즈베리파이 기반 엣지 노드에서 수집 → 특징 추출 → MQTT 퍼블리시 → 머신러닝 추론까지의 통합 파이프라인.

## 핵심 아키텍처 (필독!)

### 데이터 흐름
```
[센서 수집] → [윈도우 큐잉] → [MQTT 발행] → [추론 워커 구독] → [특징 추출] → [모델 추론] → [결과 발행]
```

**주요 컴포넌트:**
- **`src/sensor_final.py`**: DHT11, ADS1115(진동/소리), MPU6050(가속도/자이로), BMP085 센서 읽기 → 샘플링 윈도우 누적 → `factory/inference/windows/{sensor_type}` 토픽으로 MQTT 발행. **중요:** 센서별 10초 윈도우(5초 오버래핑) 설정됨 (`WINDOW_SIZE=10.0`, `WINDOW_OVERLAP=5.0` in `inference_interface.py`).
- **`src/inference_worker.py`**: MQTT 윈도우 구독 → V17 특징 추출 → MLPClassifier/IsolationForest 추론 → 결과 발행.
- **`src/feature_extractor.py`**: V17 특징 추출 (3축 벡터 스칼라화 + PCA) → 총 23개 특징 (이전: 77개).
- **`models/mlp_classifier.onnx`**: 3단계(Normal/Yellow/Red) 분류 모델 (ONNX, CPU 실행).
- **`config/sensor_config.yaml`**: MQTT 브로커, I2C 주소, 샘플링 주파수.

### 메시지 스키마 (via `inference_interface.py`)
```python
WindowMessage(sensor_type="accel_gyro", sampling_rate_hz=16.0,
             window_fields={"fields_accel_x": [...], ...}, timestamp_ns=...)
InferenceResultMessage(sensor_type="accel_gyro", model_name="mlp_classifier",
                      predictions={"class": 0, "anomaly_score": 0.15}, ...)
```

## 프로젝트 특이 패턴

### 1. 특징 추출 V17 (현재, 최적화)
- **벡터 스칼라화 + PCA**: 3축 데이터 → 1개 주축 → 특징 15개 (V16: 23개)
- **FFT 1회 호출** (이전: 6회) → 60-65% 속도 개선
- 코드: `src/feature_extractor.py` 의 `FEATURE_CONFIG_V17` dict & `extract_features_v17()` 함수
- **주의**: `inference_worker.py`의 모델 입력 15개 특징 기대 → 불일치 시 실패

### 2. MQTT 토픽 규칙 (엄격)
```
factory/inference/windows/{sensor_type}          # Publisher → Worker
factory/inference/results/{sensor_type}          # Worker → 센서별
factory/inference/results/{sensor_type}/{model}  # Worker → 모델별
```
**중요**: `inference_interface.py`의 `WINDOW_TOPIC_ROOT`, `RESULT_TOPIC_ROOT` 변경 시 publisher + worker 재시작 필수.

### 3. 나노초 타임스탐프 표준화
- `current_timestamp_ns()` 호출 (Python 3.7+ `time.time_ns()` 또는 fallback)
- 모든 MQTT 메시지에 포함 → 윈도우/결과 간 추적

### 4. 하드웨어 옵션 처리 (try/except)
```python
try:
    import Adafruit_BMP.BMP085 as BMP085
    HAS_BMP = True
except:
    HAS_BMP = False  # Windows/로컬 개발: 압력 = 0
```
- **Windows 개발**: 압력 필드 항상 0 → 테스트 시 반영 필수

### 5. 우아한 종료 패턴
```python
stop_flag = False
def signal_handler(sig, frame):
    global stop_flag
    stop_flag = True
signal.signal(signal.SIGINT, signal_handler)
```

## 개발 워크플로우

### 로컬 (하드웨어 없음)
1. `pip install -r requirements-core.txt`
2. `python dummies/data_collector.py` (모킹 센서 발행)
3. `python src/inference_worker.py` (워커)
4. `mosquitto_sub -h localhost -t "factory/inference/results/#"` (결과 확인)

### 라즈베리파이
```bash
sudo python3 src/sensor_final.py &
python3 src/inference_worker.py &
```

### 전체 실행 (Windows)
```bash
python run_all.py  # 3개 터미널 자동 시작
```

## 모델 & 스케일러

### 경로 규칙
- **모델**: `models/mlp_classifier.onnx` (ONNX), `.pth`/`.pt` (PyTorch legacy)
- **스케일러**: `models/scaler_if.joblib`, `scaler_mlp.pkl`
- **메타**: `models/training_summary.json` (입력 특징 수 등)

### 업데이트
1. 모델 학습 → `.onnx` + `.pkl` 저장
2. `python scripts/resave_models.py` (통합 로드/재저장)
3. `inference_worker.py` 재시작

## 의존성

### 핵심
`paho-mqtt`, `numpy`, `scipy`, `scikit-learn`, `onnxruntime`, `joblib`, `pandas`

### 라즈베리파이만
`adafruit-circuitpython-dht`, `adafruit-circuitpython-ads1x15`, `adafruit-bmp`, `smbus2`, `RPi.GPIO`

## 테스트
```bash
pytest tests/test_inference_flow.py -v
mosquitto_sub -h localhost -t "factory/inference/windows/#" -v | python -m json.tool
```

## 문제 해결

| 증상 | 원인 | 해결 |
|------|------|------|
| `ModuleNotFoundError: onnxruntime` | ONNX 미설치 | `pip install onnxruntime` |
| `ValueError: input size mismatch` | 특징 15개 ≠ 모델 입력 | `training_summary.json` 확인; 모델 재학습 |
| MQTT 연결 실패 | 브로커 미실행 | `sudo systemctl start mosquitto` |
| 모든 센서값 0 (Windows) | 하드웨어 import 실패 | 예상된 동작; 모킹 센서 사용 |
| `permission denied /dev/i2c-1` | I2C 권한 부족 | `sudo usermod -aG i2c $USER` (RPi)
