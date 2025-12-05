"""
추론 결과 테스트 발행기

웹 알림 테스트용으로 임의의 Isolation Forest + MLP 결과를 발행합니다.

토픽: factory/inference/results/{sensor_type}/{model_name}
형식: Telegraf mqtt_consumer 호환 JSON

사용법:
  python src/test_result_publisher.py
"""

import json
import random
import time
import signal
import sys

import paho.mqtt.client as mqtt

# ── 설정 ──
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
RESULT_TOPIC_BASE = "factory/inference/results"
SENSOR_TYPE = "accel_gyro"

# 발행 간격 (초)
PUBLISH_INTERVAL = 5.0

# 시나리오 모드: "red" = 적색 경보, "yellow" = 황색 경보, "normal" = 정상, "random" = 랜덤
SCENARIO_MODE = "red"

stop_flag = False

def current_timestamp_ns() -> int:
    try:
        return int(time.time_ns())
    except AttributeError:
        return int(time.time() * 1e9)

def signal_handler(sig, frame):
    global stop_flag
    print("\n[INFO] 종료 중...")
    stop_flag = True

def create_client():
    client_id = f"test_result_pub_{int(time.time())}"
    try:
        return mqtt.Client(client_id=client_id)
    except Exception:
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id=client_id)

def generate_iforest_result():
    """시나리오 2: 적색 경보 - 이상탐지 결과 (항상 이상)"""
    # 적색 경보: 이상탐지 점수가 낮음 (이상 상태)
    anomaly_score = random.uniform(-0.6, -0.3)  # 음수 = 이상
    prediction = -1
    is_anomaly = True
    
    return {
        "kind": "inference_result",
        "sensor_type": SENSOR_TYPE,
        "model_name": "isolation_forest",
        "timestamp_ns": current_timestamp_ns(),
        "iforest_score": round(anomaly_score, 4),
        "iforest_raw_score": round(anomaly_score - 0.1, 4),
        "iforest_prediction": prediction,
        "is_anomaly": is_anomaly,
    }

def generate_mlp_result():
    """시나리오 2: 적색 경보 - MLP가 항상 red 예측"""
    # 적색 경보: red 클래스 확률이 높음
    probs = [
        random.uniform(0.05, 0.12),   # normal: 5-12%
        random.uniform(0.08, 0.18),   # yellow: 8-18%
        random.uniform(0.70, 0.87),   # red: 70-87%
    ]
    
    # 정규화
    total = sum(probs)
    probs = [p / total for p in probs]
    
    predicted_class = 2  # red
    predicted_label = "red"
    
    return {
        "kind": "inference_result",
        "sensor_type": SENSOR_TYPE,
        "model_name": "mlp_classifier",
        "timestamp_ns": current_timestamp_ns(),
        "mlp_predicted_class": predicted_class,
        "mlp_predicted_label": predicted_label,
        "mlp_confidence": round(probs[2], 4),
        "mlp_prob_normal": round(probs[0], 4),
        "mlp_prob_yellow": round(probs[1], 4),
        "mlp_prob_red": round(probs[2], 4),
    }

def main():
    global stop_flag
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    client = create_client()
    
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        print(f"[INFO] MQTT 브로커 연결됨: {MQTT_BROKER}:{MQTT_PORT}")
    except Exception as e:
        print(f"[ERROR] 브로커 연결 실패: {e}")
        sys.exit(1)
    
    client.loop_start()
    
    print(f"[INFO] 토픽 베이스: {RESULT_TOPIC_BASE}/{SENSOR_TYPE}/*")
    print(f"[INFO] 발행 간격: {PUBLISH_INTERVAL}초")
    print("[INFO] Ctrl+C로 종료\n")
    
    count = 0
    while not stop_flag:
        count += 1
        
        # Isolation Forest 결과 발행
        iforest_result = generate_iforest_result()
        iforest_topic = f"{RESULT_TOPIC_BASE}/{SENSOR_TYPE}/isolation_forest"
        client.publish(iforest_topic, json.dumps(iforest_result))
        
        status = "⚠️ 이상" if iforest_result["is_anomaly"] else "✅ 정상"
        print(f"[{count}] IFOREST: score={iforest_result['iforest_score']:.4f} {status}")
        
        # MLP Classifier 결과 발행
        mlp_result = generate_mlp_result()
        mlp_topic = f"{RESULT_TOPIC_BASE}/{SENSOR_TYPE}/mlp_classifier"
        client.publish(mlp_topic, json.dumps(mlp_result))
        
        label = mlp_result["mlp_predicted_label"]
        emoji = {"normal": "🟢", "yellow": "🟡", "red": "🔴"}.get(label, "⚪")
        print(f"[{count}] MLP: {emoji} {label} (신뢰도: {mlp_result['mlp_confidence']:.2%})")
        print()
        
        time.sleep(PUBLISH_INTERVAL)
    
    client.loop_stop()
    client.disconnect()
    print("[INFO] 종료됨")

if __name__ == "__main__":
    main()
