# MOBY Edge Sensor Publisher

간단 소개
- 이 저장소는 라즈베리파이 기반 엣지 노드용 센서 수집 및 MQTT 퍼블리셔 스크립트 모음입니다.
- 주요 스크립트: `src/sensor_final.py` (통합 퍼블리셔 + 비동기 인퍼런스 워커)

빠른 시작
1. (권장) 가상환경 생성 및 활성화:
```
python3 -m venv .venv
source .venv/bin/activate
```
2. 필수 패키지 설치:
```
pip install -r requirements.txt
# 하드웨어 의존성만 라즈베리파이에 설치하려면:
pip install -r requirements-hw.txt
```
3. 실행:
```
python src/sensor_final.py
```

인퍼런스 요약 (간단)
- 트리거: MPU6050(IMU) 윈도우가 채워지면 윈도우를 큐에 넣고 비동기 워커가 처리합니다.
- 처리 흐름: 윈도우 → `extract_features()` → `scaler.transform()` → `model.score_samples()` / `model.predict()` → MQTT 퍼블리시
- 발행 토픽: `factory/sensor/<sensor_type>/inference` (예: `factory/sensor/accel_gyro/inference`)
- 페이로드: 원래 센서 페이로드에 `fields.anomaly_score` 와 `fields.anomaly_label` 추가

주의 및 권장 사항
- 재전송 버퍼(`BUFFER_DIR`)에 저장된 페이로드는 재전송 시 원래의 `/inference` 토픽이 아닌 기본 토픽으로 발행될 수 있으므로, 정확한 재전송을 위해 `publish_topic` 필드 저장을 권장합니다.
- 모델/스케일 버전 불일치 경고가 발생할 수 있으니(직렬화 시 사용된 scikit-learn 버전과 실행 환경 버전 불일치) 모델을 현재 환경에 맞게 재저장하거나(권장) venv의 scikit-learn 버전을 모델의 버전으로 맞추세요.

추가 문서
- 인퍼런스 동작의 자세한 설명은 `docs/inference.md`에 있습니다.

문의 및 다음 단계
- README에 더 자세한 실행/디버깅 절차를 추가해 드릴까요? (예: 테스트 모드, 더미 데이터 주입 방법)
## 데이터 신뢰성/버퍼링

MQTT 발행 실패 시 센서별 payload를 `/home/wise/deployment/data/buffer/`에 `.npy` 파일로 자동 저장합니다.
- 파일명: `{sensor_type}_{timestamp_ns}.npy` (numpy object array)
- 루프 시작 시 버퍼 폴더 내 모든 `.npy` 파일을 MQTT로 재전송, 성공 시 파일 삭제
- 버퍼 파일이 100개를 초과하면 오래된 파일부터 삭제해 100개만 유지합니다.

이 기능은 별도 설정 없이 자동 동작하며, 데이터 유실 방지 및 신뢰성 확보에 도움이 됩니다.

# MOBY Edge Sensor Scripts

간단한 요약: 이 저장소는 라즈베리파이에서 여러 센서를 읽어 MQTT로 퍼블리시하는 독립형 Python 스크립트 모음입니다.

주요 파일
- `sensor_final.py`  : 통합 퍼블리셔 + 팬 제어(TB6612)를 포함한 데모(실제 실행용).
- `data_collector.py`: ADS1115, DHT11, MPU6050, BMP180 등을 읽어 `factory/sensor/<type>` 토픽으로 전송.
- `sensor.py`, `DHT11.py`: DHT11 사용 예제 및 드라이버 샘플.
- `dht11_mqtt.py`, `IR_mqtt.py`: MQTT 퍼블리시/구독 예제.

핵심 패턴
- 각 파일은 독립 실행형 스크립트이며 `if __name__ == "__main__": main()` 패턴을 따릅니다.
- 하드코드된 구성(브로커, 토픽, I2C 주소, 핀)은 파일 상단에 위치합니다. 수정 시 해당 상단 섹션을 확인하세요.
- 하드웨어 의존 라이브러리는 try/except로 optional 처리됩니다(예: `Adafruit_BMP`).
- 안전 종료: 전역 `stop_flag`와 `signal(SIGINT/SIGTERM)` 처리로 루프 종료 및 정리 수행.

의존성(권장)
- 라즈베리파이 환경에서 필요할 가능성이 높은 패키지:

```
pip3 install adafruit-circuitpython-dht adafruit-circuitpython-ads1x15 smbus2 paho-mqtt Adafruit-BMP RPi.GPIO
```

실행 예시
- 라즈베리파이에서 GPIO/I2C 접근이 필요하므로 보통 `sudo`로 실행:

```bash
sudo python3 sensor_final.py
# 또는
python3 data_collector.py
```

로컬(하드웨어 없음) 개발
- 하드웨어 라이브러리가 없거나 센서가 없을 경우 import 실패가 발생할 수 있습니다. 그런 경우 테스트용 브로커(`mosquitto`)를 설치하고 센서 읽기 부분을 모킹하거나 try/except 분기로 처리하세요.

MQTT 통합 요약
- 기본 브로커: `BROKER = "localhost", PORT = 1883` (파일 상단).
- 토픽 관례: `factory/sensor/<type>` (예: `factory/sensor/dht11`).
- 페이로드: JSON 구조, `timestamp_ns`(나노초) 포함. 예시는 `.github/copilot-instructions.md`에 포함.

테스트 팁
- 브로커가 로컬에 없으면 간단히 `mosquitto`를 설치해 테스트하세요.

추가 안내
- 더 자세한 환경(사용 중인 Raspbian 버전, Python 버전, 테스트 브로커 유무)을 알려주시면 README에 구체적 설치/테스트 절차를 추가하겠습니다.

---
파일에서 발견한 관례와 예시만을 바탕으로 작성했습니다. 필요한 추가 항목(예: CI, 도커, 모킹 샘플)을 알려주시면 바로 추가하겠습니다.
