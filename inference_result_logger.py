import csv
import json
import time
import os
import paho.mqtt.client as mqtt
from datetime import datetime

# ==========================================
# 설정
# ==========================================
import os
BROKER = os.getenv("MQTT_BROKER", "192.168.80.208")
PORT = int(os.getenv("MQTT_PORT", "1883"))
TOPIC_ROOT = "factory/inference/results/#"  # 모든 추론 결과 구독
SAVE_DIR = "./inference_results"            # CSV 저장 경로

# 파일 핸들러들을 관리할 딕셔너리
file_handlers = {}

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def get_file_handler(sensor_type, model_name):
    """센서 타입과 모델별로 날짜가 적힌 CSV 파일을 엽니다."""
    today_str = datetime.now().strftime("%Y%m%d")
    filename = f"{sensor_type}_{model_name}_{today_str}.csv"
    filepath = os.path.join(SAVE_DIR, filename)

    # 이미 열려있고 날짜가 같다면 그대로 반환
    key = f"{sensor_type}_{model_name}"
    if key in file_handlers:
        handler = file_handlers[key]
        if handler['filepath'] == filepath:
            return handler
        else:
            # 날짜가 바뀌었으면 기존 파일 닫기
            handler['file'].close()

    # 새 파일 열기 (append 모드)
    is_new_file = not os.path.exists(filepath)
    f = open(filepath, "a", newline='', encoding='utf-8')
    writer = csv.writer(f)
    
    file_handlers[key] = {
        'file': f, 
        'writer': writer, 
        'filepath': filepath, 
        'header_written': not is_new_file
    }
    return file_handlers[key]

def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT Broker (Result: {rc})")
    print(f"Subscribing to {TOPIC_ROOT} for inference result logging...")
    client.subscribe(TOPIC_ROOT)

def on_message(client, userdata, msg):
    try:
        # 페이로드 디코딩
        payload_str = msg.payload.decode("utf-8")
        data = json.loads(payload_str)
        
        # 데이터 파싱
        sensor_type = data.get("sensor_type", "unknown")
        model_name = data.get("model_name", "unknown")
        score = data.get("score")
        timestamp_ns = data.get("timestamp_ns", time.time_ns())
        context_payload = data.get("context_payload", {})
        context_fields = context_payload.get("fields", {})

        # CSV 핸들러 가져오기
        handler = get_file_handler(sensor_type, model_name)
        writer = handler['writer']

        # 헤더 작성 (파일이 새로 생성되었고 아직 헤더를 안 썼다면)
        if not handler['header_written']:
            headers = [
                "timestamp_ns",
                "sensor_type",
                "model_name",
                "score"
            ]
            # context_fields의 모든 키를 헤더에 추가
            headers.extend(sorted(context_fields.keys()))
            writer.writerow(headers)
            handler['header_written'] = True
            handler['file'].flush()

        # 데이터 작성
        row = [
            timestamp_ns,
            sensor_type,
            model_name,
            score if score is not None else ""
        ]
        # context_fields의 값을 헤더 순서대로 추가
        sorted_keys = sorted(context_fields.keys())
        row.extend([context_fields.get(key, "") for key in sorted_keys])
        
        writer.writerow(row)
        handler['file'].flush()
        
        print(
            f"[LOGGED] {sensor_type} | {model_name} | score={score}"
        )

    except Exception as e:
        print(f"Error logging inference result: {e}")

def main():
    # Paho MQTT v2.0 호환성 처리
    try:
        client = mqtt.Client(
            client_id="Inference_Result_Logger_Service",
            callback_api_version=mqtt.CallbackAPIVersion.VERSION1
        )
    except AttributeError:
        # Paho MQTT v1.x 버전
        client = mqtt.Client("Inference_Result_Logger_Service")

    client.on_connect = on_connect
    client.on_message = on_message

    print(f"Connecting to {BROKER}:{PORT}...")
    client.connect(BROKER, PORT, 60)
    
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        client.disconnect()

if __name__ == "__main__":
    main()
