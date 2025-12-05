#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import signal
import threading
import json
import collections
import numpy as np
from dotenv import load_dotenv
import RPi.GPIO as GPIO
import paho.mqtt.client as mqtt
from datetime import datetime, timedelta

# Load .env file
load_dotenv()

# ==============================
# Config
# ==============================
AIN1 = 27
AIN2 = 22
PWMA = 18
STBY = 23

PWM_FREQ = 100     # Hz
DUTY = 75          # 0..100 (%)

stop_flag = False

# -----------------------------
# [PREDICTION CONFIG] 예지보전 설정 (단일 기준)
# -----------------------------
# 11/25일 초기 안정 구간(4583ms) 대비 +15% 지점을 '고장 위험'으로 정의
# 이 값을 넘으면 즉시 CRITICAL 상태로 판단
FAILURE_THRESHOLD_MS = 5000

# 추세 분석을 위한 윈도우 크기
TREND_WINDOW_SIZE = 50       

# -----------------------------
# IR sensor + MQTT
# -----------------------------
IR_PIN = 17
DEAD_TIME_MS = 200
AVG_WINDOW = 10
PRINT_EVERY = 1
MQTT_BROKER = os.getenv("MQTT_BROKER", "192.168.80.208")
MQTT_PORT = 1883
MQTT_TOPIC = "factory/conveyor/ir"
MQTT_CLIENT_ID = "IR_Conveyor_Sensor"

mqtt_client = None
last_hit_ns = None
dead_until_ns = 0
cycle_times_ms = []
cycle_count = 0
ir_thread = None

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
    try:
        status = msg.get('health')
        # 심플해진 상태 아이콘: 정상(초록) vs 위험(빨강)
        icon = "🟢" if status == "NORMAL" else "🔴"
        
        log_str = f"[MQTT] {icon} {status} | Cycle={msg['cycles']} | Avg={msg['avg_cycle_ms']}ms"
        
        if msg.get('rul_hours'):
            log_str += f" | 🕒 RUL: {msg['rul_hours']}h ({msg['fail_time']})"
        
        print(log_str)
            
    except Exception:
        pass

    if mqtt_client:
        try:
            mqtt_client.publish(MQTT_TOPIC, json.dumps(msg))
        except Exception:
            pass

def predict_failure(current_avg_ms):
    """
    현재 이동평균 값을 바탕으로 '고장 위험(CRITICAL)' 도달 시점을 예측합니다.
    """
    # 데이터 부족 시 스킵
    if len(trend_buffer) < 10:
        return "CALCULATING", None, 0.0

    # 1. 상태 진단 (심플하게 2단계)
    if current_avg_ms >= FAILURE_THRESHOLD_MS:
        health = "CRITICAL"  # 위험/고장 임박
    else:
        health = "NORMAL"    # 정상

    # 2. 기울기 계산 (Numpy Linear Regression)
    y = np.array(trend_buffer)
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)

    # 3. 잔존 수명(RUL) 예측
    # 목표 지점(5270ms)까지 얼마나 남았나?
    rul_cycles = None
    
    if slope > 0.01:
        remaining_ms = FAILURE_THRESHOLD_MS - current_avg_ms
        if remaining_ms > 0:
            rul_cycles = int(remaining_ms / slope)
        else:
            rul_cycles = 0 # 이미 도달함
    else:
        rul_cycles = 999999 # 안정적임

    return health, rul_cycles, round(slope, 5)

def record_hit(t_ns):
    global last_hit_ns, dead_until_ns, cycle_count, cycle_times_ms
    
    if t_ns < dead_until_ns: return
    if last_hit_ns is None:
        last_hit_ns = t_ns
        dead_until_ns = t_ns + DEAD_TIME_MS * 1_000_000
        return

    dt_ms = (t_ns - last_hit_ns) / 1_000_000.0
    last_hit_ns = t_ns
    dead_until_ns = t_ns + DEAD_TIME_MS * 1_000_000

    if dt_ms < DEAD_TIME_MS * 1.2: return

    cycle_count += 1
    cycle_times_ms.append(dt_ms)
    
    if len(cycle_times_ms) > AVG_WINDOW:
        cycle_times_ms = cycle_times_ms[-AVG_WINDOW:]

    if cycle_count % PRINT_EVERY == 0:
        avg_ms = sum(cycle_times_ms) / len(cycle_times_ms) if cycle_times_ms else 0.0
        
        if avg_ms > 0:
            trend_buffer.append(avg_ms)
        
        health_status, rul_cycles, slope = predict_failure(avg_ms)
        
        rul_hours = None
        fail_time_str = None
        
        if rul_cycles is not None and rul_cycles != 999999:
            seconds_left = rul_cycles * (avg_ms / 1000.0)
            rul_hours = round(seconds_left / 3600.0, 2)
            fail_dt = datetime.now() + timedelta(seconds=seconds_left)
            fail_time_str = fail_dt.strftime("%Y-%m-%d %H:%M:%S")
            
        msg = {
            "cycles": cycle_count,
            "last_cycle_ms": round(dt_ms, 2),
            "avg_cycle_ms": round(avg_ms, 2),
            "timestamp_ns": t_ns,
            "health": health_status,       # NORMAL vs CRITICAL
            "slope": slope,
            "rul_cycles": rul_cycles,
            "rul_hours": rul_hours,
            "fail_time": fail_time_str
        }
        _publish_ir(msg)

def ir_polling_loop():
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
                if prev == 0 and cur == 1: record_hit(now_ns())
            else:
                if prev == 1 and cur == 0: record_hit(now_ns())
            prev = cur
            time.sleep(0.001)
    except Exception: return

def start_ir_thread():
    global ir_thread
    if ir_thread is None:
        ir_thread = threading.Thread(target=ir_polling_loop, daemon=True)
        ir_thread.start()

def handle_sigint(sig, frame):
    global stop_flag
    print("\n[MOTOR] Interrupt received. Stopping motor...")
    stop_flag = True

signal.signal(signal.SIGINT, handle_sigint)

def init_gpio_bcm():
    GPIO.setwarnings(False)
    mode = GPIO.getmode()
    if mode is not None and mode != GPIO.BCM: GPIO.cleanup()
    try: GPIO.setmode(GPIO.BCM)
    except ValueError: GPIO.cleanup(); GPIO.setmode(GPIO.BCM)
    GPIO.setup(AIN1, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(AIN2, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(PWMA, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(STBY, GPIO.OUT, initial=GPIO.LOW)

def main():
    init_gpio_bcm()
    init_mqtt()
    start_ir_thread()
    pwm = GPIO.PWM(PWMA, PWM_FREQ)
    pwm.start(0)

    print(f"[SYSTEM] Simplified RUL System Started.")
    print(f"[CONFIG] Critical Threshold: {FAILURE_THRESHOLD_MS}ms")
    
    GPIO.output(STBY, GPIO.HIGH)
    GPIO.output(AIN1, GPIO.HIGH)
    GPIO.output(AIN2, GPIO.LOW)
    pwm.ChangeDutyCycle(DUTY)

    try:
        while not stop_flag: time.sleep(1.0)
    finally:
        pwm.ChangeDutyCycle(0)
        pwm.stop()
        GPIO.cleanup()
        if mqtt_client: mqtt_client.loop_stop(); mqtt_client.disconnect()
        print("[MOTOR] Clean exit.")

if __name__ == "__main__":
    main()