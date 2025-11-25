#!/usr/bin/env python3
# -*- coding: us-ascii -*-

import time
import signal
import threading
import json
import collections
import numpy as np  # pip install numpy 필요
import RPi.GPIO as GPIO
import paho.mqtt.client as mqtt

# ==============================
# Config
# ==============================
AIN1 = 27
AIN2 = 22
PWMA = 18
STBY = 23

PWM_FREQ = 100     # Hz
DUTY = 75          # 0..100 (%), set your steady speed here

stop_flag = False

# -----------------------------
# Prediction Config (NEW)
# -----------------------------
# 분석 결과를 바탕으로 설정한 임계값
WARNING_THRESHOLD_MS = 5000  # 주의 단계 (Fluctuating 시작)
FAILURE_THRESHOLD_MS = 6000  # 고장 판단 (멈춤/심각한 저하)
TREND_WINDOW_SIZE = 50       # 추세를 계산할 최근 데이터 개수 (이동평균 값 기준)

# -----------------------------
# IR sensor + MQTT
# -----------------------------
IR_PIN = 17
DEAD_TIME_MS = 200
AVG_WINDOW = 10
PRINT_EVERY = 1
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "factory/conveyor/ir"
MQTT_CLIENT_ID = "IR_Conveyor_Sensor"

mqtt_client = None
last_hit_ns = None
dead_until_ns = 0
cycle_times_ms = []
cycle_count = 0
ir_thread = None

# 예측을 위한 버퍼 (deque: 꽉 차면 오래된 것 자동 삭제)
trend_buffer = collections.deque(maxlen=TREND_WINDOW_SIZE)

def now_ns():
    return time.time_ns()

def init_mqtt():
    global mqtt_client
    try:
        mqtt_client = mqtt.Client(client_id=MQTT_CLIENT_ID)
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
    except Exception:
        mqtt_client = None

def _publish_ir(msg: dict):
    # Log the outgoing MQTT payload to the terminal
    try:
        print(f"[MQTT] Status={msg.get('health')} | RUL={msg.get('rul_cycles')} | Slope={msg.get('slope')}")
    except Exception:
        pass

    if mqtt_client:
        try:
            mqtt_client.publish(MQTT_TOPIC, json.dumps(msg))
        except Exception:
            pass

def predict_failure(current_avg_ms):
    """
    현재의 이동평균 값들을 바탕으로 고장 시점을 예측합니다.
    Returns: (health_status, cycles_to_failure, current_slope)
    """
    # 데이터가 충분하지 않으면 계산 스킵
    if len(trend_buffer) < 10:
        return "CALCULATING", None, 0.0

    # 1. 상태 진단 (Threshold Check)
    health = "NORMAL"
    if current_avg_ms >= FAILURE_THRESHOLD_MS:
        health = "CRITICAL"
    elif current_avg_ms >= WARNING_THRESHOLD_MS:
        health = "WARNING"

    # 2. 기울기 계산 (Linear Regression using Numpy)
    # y = current_avg_ms history, x = indices (0, 1, 2...)
    y = np.array(trend_buffer)
    x = np.arange(len(y))
    
    # 1차 함수(직선) 피팅: [기울기, 절편] 반환
    slope, intercept = np.polyfit(x, y, 1)

    # 3. 잔존 수명(RUL) 예측
    # 기울기가 양수(속도가 느려짐)일 때만 예측 가능
    rul_cycles = None
    if slope > 0.001:  # 노이즈 고려하여 아주 작은 기울기는 무시
        remaining_ms = FAILURE_THRESHOLD_MS - current_avg_ms
        if remaining_ms > 0:
            rul_cycles = int(remaining_ms / slope)
        else:
            rul_cycles = 0
    else:
        # 기울기가 음수(빨라짐)거나 0이면 안정적인 상태
        rul_cycles = 999999 # Infinite/Safe

    return health, rul_cycles, round(slope, 5)

def record_hit(t_ns):
    """
    Records a valid sensor hit and publishes MQTT message with timestamp.
    """
    global last_hit_ns, dead_until_ns, cycle_count, cycle_times_ms
    
    # Debounce check
    if t_ns < dead_until_ns:
        return

    # First hit initialization
    if last_hit_ns is None:
        last_hit_ns = t_ns
        dead_until_ns = t_ns + DEAD_TIME_MS * 1_000_000
        return

    # Calculate time delta
    dt_ms = (t_ns - last_hit_ns) / 1_000_000.0
    last_hit_ns = t_ns
    dead_until_ns = t_ns + DEAD_TIME_MS * 1_000_000

    # Filter out noise (too fast hits)
    if dt_ms < DEAD_TIME_MS * 1.2:
        return

    cycle_count += 1
    cycle_times_ms.append(dt_ms)
    
    # AVG_WINDOW for smoothing noise (Short-term)
    if len(cycle_times_ms) > AVG_WINDOW:
        cycle_times_ms = cycle_times_ms[-AVG_WINDOW:]

    if cycle_count % PRINT_EVERY == 0:
        # Calculate Moving Average (Noise Filtered)
        avg_ms = sum(cycle_times_ms) / len(cycle_times_ms) if cycle_times_ms else 0.0
        
        # --- [PREDICTION LOGIC START] ---
        # Add smoothed value to trend buffer
        if avg_ms > 0:
            trend_buffer.append(avg_ms)
        
        # Predict Health & RUL
        health_status, rul, slope = predict_failure(avg_ms)
        # --- [PREDICTION LOGIC END] ---

        msg = {
            "cycles": cycle_count,
            "last_cycle_ms": round(dt_ms, 2),
            "avg_cycle_ms": round(avg_ms, 2),
            "timestamp_ns": t_ns,
            # Added Prediction Fields
            "health": health_status,       # NORMAL, WARNING, CRITICAL
            "rul_cycles": rul,             # 고장까지 남은 횟수 (예측값)
            "slope": slope                 # 현재 성능 저하 속도 (ms/cycle)
        }
        _publish_ir(msg)

def ir_polling_loop():
    # ensure pin mode independent from main init
    try:
        GPIO.setup(IR_PIN, GPIO.IN)
        vals = []
        t0 = time.time()
        while time.time() - t0 < 0.3:
            vals.append(GPIO.input(IR_PIN))
            time.sleep(0.01)
        idle = 1 if (vals and sum(vals) >= len(vals)/2.0) else 0
        pud = GPIO.PUD_DOWN if idle == 0 else GPIO.PUD_UP
        edge_str = "RISING" if idle == 0 else "FALLING"
        GPIO.setup(IR_PIN, GPIO.IN, pull_up_down=pud)
        prev = GPIO.input(IR_PIN)
        while not stop_flag:
            cur = GPIO.input(IR_PIN)
            if edge_str == "RISING":
                if prev == 0 and cur == 1:
                    record_hit(now_ns())
            else:
                if prev == 1 and cur == 0:
                    record_hit(now_ns())
            prev = cur
            time.sleep(0.001)
    except Exception:
        return

def start_ir_thread():
    global ir_thread
    if ir_thread is None:
        ir_thread = threading.Thread(target=ir_polling_loop, daemon=True)
        ir_thread.start()

# ==============================
# Graceful shutdown
# ==============================
def handle_sigint(sig, frame):
    global stop_flag
    print("\n[MOTOR] Interrupt received. Stopping motor...")
    stop_flag = True

signal.signal(signal.SIGINT, handle_sigint)

# ==============================
# GPIO init (robust)
# ==============================
def init_gpio_bcm():
    GPIO.setwarnings(False)
    mode = GPIO.getmode()
    if mode is not None and mode != GPIO.BCM:
        GPIO.cleanup()
    try:
        GPIO.setmode(GPIO.BCM)
    except ValueError:
        GPIO.cleanup()
        GPIO.setmode(GPIO.BCM)

    GPIO.setup(AIN1, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(AIN2, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(PWMA, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(STBY, GPIO.OUT, initial=GPIO.LOW)

# ==============================
# Main
# ==============================
def main():
    init_gpio_bcm()
    # start IR monitor and MQTT
    init_mqtt()
    start_ir_thread()
    pwm = GPIO.PWM(PWMA, PWM_FREQ)
    pwm.start(0)

    print("[MOTOR] Running continuously. Ctrl+C to stop.")
    GPIO.output(STBY, GPIO.HIGH)

    # Forward direction (fixed)
    GPIO.output(AIN1, GPIO.HIGH)
    GPIO.output(AIN2, GPIO.LOW)

    # Apply steady speed
    pwm.ChangeDutyCycle(DUTY)

    try:
        # Keep running at constant speed
        while not stop_flag:
            time.sleep(1.0)
    finally:
        pwm.ChangeDutyCycle(0)
        pwm.stop()
        GPIO.output(AIN1, GPIO.LOW)
        GPIO.output(AIN2, GPIO.LOW)
        GPIO.output(STBY, GPIO.LOW)
        GPIO.cleanup()
        # stop IR mqtt thread and cleanup MQTT
        try:
            if mqtt_client:
                mqtt_client.loop_stop()
                mqtt_client.disconnect()
        except Exception:
            pass
        print("[MOTOR] Clean exit.")

if __name__ == "__main__":
    main()