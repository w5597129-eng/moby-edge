import csv
import json
import time
import os
from pathlib import Path
import paho.mqtt.client as mqtt
from datetime import datetime

# .env 파일에서 환경변수 로드
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

# ==========================================
# 설정
# ==========================================
BROKER = os.getenv("MQTT_BROKER", "192.168.80.208")
PORT = int(os.getenv("MQTT_PORT", "1883"))
TOPIC_ROOT = "factory/sensor/#"  # 모든 센서 데이터 구독
SAVE_DIR = "./sensor_data"       # CSV 저장 경로

# 파일 핸들러들을 관리할 딕셔너리
file_handlers = {}

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def get_file_handler(sensor_type):
    """센서 타입별로 날짜가 적힌 CSV 파일을 엽니다."""
    today_str = datetime.now().strftime("%Y%m%d")
    filename = f"{sensor_type}_{today_str}.csv"
    filepath = os.path.join(SAVE_DIR, filename)

    # 이미 열려있고 날짜가 같다면 그대로 반환
    if sensor_type in file_handlers:
        handler = file_handlers[sensor_type]
        if handler['filepath'] == filepath:
            return handler
        else:
            # 날짜가 바뀌었으면 기존 파일 닫기
            handler['file'].close()

    # 새 파일 열기 (append 모드)
    is_new_file = not os.path.exists(filepath)
    # 버퍼링 없이 바로 쓰려면 buffering=0 (바이너리 모드 필요) 또는 flush() 호출
    # 여기서는 텍스트 모드이므로 flush()를 자주 호출하는 방식으로 처리
    f = open(filepath, "a", newline='', encoding='utf-8')
    writer = csv.writer(f)
    
    file_handlers[sensor_type] = {'file': f, 'writer': writer, 'filepath': filepath, 'header_written': not is_new_file}
    return file_handlers[sensor_type]

def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT Broker (Result: {rc})")
    print(f"Subscribing to {TOPIC_ROOT} for CSV logging...")
    client.subscribe(TOPIC_ROOT)

def on_message(client, userdata, msg):
    try:
        # 페이로드 디코딩
        payload_str = msg.payload.decode("utf-8")
        data = json.loads(payload_str) # json.load(msg.payload) 대신 json.loads 사용
        
        # 데이터 파싱
        sensor_type = data.get("sensor_type", "unknown")
        timestamp = data.get("timestamp_ns", time.time_ns())
        fields = data.get("fields", {})

        # CSV 핸들러 가져오기
        handler = get_file_handler(sensor_type)
        writer = handler['writer']

        # 헤더 작성 (파일이 새로 생성되었고 아직 헤더를 안 썼다면)
        if not handler['header_written']:
            # [Timestamp] + [Keys of fields]
            headers = ["timestamp_ns"] + list(fields.keys())
            writer.writerow(headers)
            handler['header_written'] = True
            handler['file'].flush() # 헤더 즉시 저장

        # 데이터 작성
        row = [timestamp] + list(fields.values())
        writer.writerow(row)
        
        # 실시간 저장을 위해 flush (데이터 유실 방지)
        handler['file'].flush() 

    except Exception as e:
        # JSON 형식이 아니거나, 필드가 없는 경우 무시하거나 에러 출력
        # print(f"Error saving CSV: {e}")
        pass

def main():
    # [수정된 부분] Paho MQTT v2.0 호환성 처리
    # CallbackAPIVersion.VERSION1을 사용해야 기존 콜백 함수(on_connect 등)를 그대로 쓸 수 있음
    try:
        client = mqtt.Client(client_id="CSV_Logger_Service", callback_api_version=mqtt.CallbackAPIVersion.VERSION1)
    except AttributeError:
        # Paho MQTT v1.x 버전을 쓰는 경우 (CallbackAPIVersion이 없음)
        client = mqtt.Client("CSV_Logger_Service")

    client.on_connect = on_connect
    client.on_message = on_message

    print(f"Connecting to {BROKER}:{PORT}...")
    client.connect(BROKER, PORT, 60)
    
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\nStopping CSV Logger...")
        for h in file_handlers.values():
            h['file'].close()
        print("Files closed. Bye.")

if __name__ == "__main__":
    main()