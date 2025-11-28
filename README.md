# MOBY Edge Sensor Publisher

IoT 센서 데이터를 라즈베리파이 기반 엣지 노드에서 수집해 MQTT로 퍼블리시하는 스크립트·모듈 모음입니다. 센서·모터·추론 파이프라인을 독립 실행형 파일로 제공하여, 센서가 설치된 장비에서는 `src/sensor_final.py`를, 워커·모델이 필요한 환경에서는 `src/inference_worker.py` 를 별도 터미널에서 띄워 동작시킬 수 있도록 구성했습니다.

## Repository layout
- `src/`: 센서 퍼블리셔, 추론 워커, 메시지/피처 인터페이스 및 핵심 특징 추출기(`feature_extractor` V17 포함).
- `scripts/`: 모델/스케일러 재저장(`resave_models.py`), MQTT 구독/테스트 코드, ONNX/모델 처리 보조 스크립트.
- `models/`: 학습된 ONNX/PyTorch 모델(`mlp_classifier.*`), isolation forest `joblib`, 스케일러 파일, `training_summary.json` 등의 추론 자산.
- `dummies/`: 센서 드라이버 및 샘플 퍼블리셔(예: `DHT11.py`, `bmp280.py`, `accel_gyro.py`, `data_collector.py`)로 로컬 개발·모킹에 활용.
- `config/`: `sensor_config.yaml` 등 하드웨어 매핑 및 MQTT 구성 상수를 담고 있으며, 센서 스크립트에서 직접 읽습니다.
- `sensor_data/`: 실험/로깅용 센서 데이터 저장소(로컬 처리/전처리 결과물). `data/processed/` 하위에는 머신러닝 입력용 피처 파일을 둡니다.
- 루트 스크립트들: `csv_logger.py`, `feature_extractor.py`(루트 래퍼), `motor*.py`, `IR_mqtt_pub.py` 등 실행·개발 편의용 스크립트.

## Setup
1. (권장) 가상환경 생성·활성화
```bash
python3 -m venv .venv
source .venv/bin/activate
```
2. 의존성 설치
```bash
pip install -r requirements.txt
```
- 하드웨어 지원만 필요한 경우 `pip install -r requirements-hw.txt` 를, 최소 라이브러리만 필요하면 `requirements-core.txt` 를 이용하세요.
3. 항상 리포지토리 루트(`/home/wise/python`)에서 명령을 실행해야 상대 경로 기반의 모델·스크립트를 찾습니다.

## Running key components
- **센서 퍼블리셔**: `sudo python src/sensor_final.py` (GPIO/I2C 권한 필요). 센서 출력은 `factory/sensor/<type>` 토픽으로 publish하며, MPU6050 윈도우가 찰 경우 `factory/inference/windows/<sensor_type>` 윈도우 메시지를 전송합니다.
- **Inference worker**: `python src/inference_worker.py`. MQTT로 윈도우를 구독하여 `feature_extractor.extract_features_v17` 를 우선 호출하고, fallback으로 1D 신호 기반 `extract_features` 를 사용합니다. 결과는 `factory/inference/results/<sensor_type>` 또는 모델별 토픽으로 전송됩니다.
- **모델/스케일러 갱신**: `python scripts/resave_models.py` 로 기존 모델/스케일러를 다시 저장할 수 있습니다(CPU 환경 권장).
- **motor/IR 스크립트**: `python motor_slow.py`, `python motor.py`, `python IR_mqtt_pub.py` 등은 IR 이벤트·모터 제어 테스트용이며, publish 로그를 MQTT로 내보냅니다.

## MQTT & broker testing
- 로컬 브로커가 필요하면 `mosquitto` 설치 후 실행
```bash
sudo apt update
sudo apt install -y mosquitto
sudo systemctl enable --now mosquitto
```
- 구독 예시
```bash
mosquitto_sub -h localhost -t "factory/inference/windows/#" -v
mosquitto_sub -h localhost -t "factory/inference/results/#" -v
```

## Hardware notes & development aids
- 센서 드라이버 모듈(`DHT11`, `bmp280`, `accel_gyro` 등)은 `dummies/` 에 샘플 구현이 있으니 하드웨어가 없을 경우 로컬에서 테스트하거나 모킹에 사용하세요.
- 하드웨어 관련 import 실패 시 try/except 처리된 대체 흐름이 있으므로, 로컬 개발에서는 예외 구간을 그대로 두고 `mosquitto` 나 `scripts/mqtt_subscribe_test.py`로 메시지를 publish 해 워커 로직을 검증하세요.
- GPIO/I2C 접근이 필요한 경우 보통 `sudo` 또는 적절한 그룹 권한이 필요하므로, 권한 문제 발생 시 `python` 명령으로 직접 실행하는 방식을 추천합니다.
- `config/sensor_config.yaml` 을 통해 사용 중인 센서 유형·토픽·윈도우 설정을 조정할 수 있으며, `src/sensor_final.py` 와 `src/inference_worker.py` 모두 이 설정을 참조합니다.

## Troubleshooting & tips
- GPU 관련 경고 제거: `ONNXMLPWrapper` 는 기본적으로 CPU 실행 프로바이더(`CPUExecutionProvider`)를 사용합니다. GPU를 쓰려면 `onnxruntime-gpu` 를 설치하거나 `providers` 파라미터를 직접 지정해주세요.
- 모델 호환성: V17 피처 집합이 기존 PyTorch 체크포인트(`*.pth`, `*.pt`)와 차이가 날 수 있으니, 필요하면 모델을 ONNX로 변환하거나 `scripts/resave_models.py` 로 입력 차원을 다시 저장하세요.
- `paho-mqtt` 콜백 API DeprecationWarning은 동작에는 영향을 주지 않지만, `_make_mqtt_client()` 내부에서 경고를 무시하거나 라이브러리 업그레이드를 고려하세요.
- 테스트: `pytest` 를 통해 `tests/test_inference_flow.py` 등 워커 흐름을 검증할 수 있으며, `feature_extractor` 로컬 래퍼인 `feature_extractor.py` 를 사용하면 `src` 경로 없이도 같은 API를 호출할 수 있습니다.

## Next steps & resources
- 센서(퍼블리셔)와 워커는 각각 별도 터미널(또는 systemd 서비스)에서 실행하여 같은 MQTT 브로커에 연결하도록 구성하세요.
- 추가 문서(예: systemd 서비스 예시, Dockerfile, 더미 데이터 주입 스크립트)가 필요하면 알려주시면 README에 반영합니다.

1️⃣ Windows PowerShell/CMD 사용 (추천)
→ 3개의 별개 터미널 창이 자동으로 열림
```
cd c:\Users\pinu4\project\python
.\run_all.bat
```

2️⃣ Python 통합 런처 (크로스 플랫폼)
→ 하나의 터미널에서 모든 서비스 관리
```
cd c:\Users\pinu4\project\python
python run_all.py
```

✅ 로그 통합 관리
✅ Ctrl+C로 모든 서비스 동시 종료
✅ 프로세스 모니터링
3️⃣ Linux/Raspberry Pi (bash script)
```
cd /path/to/project/python
bash run_all.sh
```

→ 백그라운드에서 모든 프로세스 실행