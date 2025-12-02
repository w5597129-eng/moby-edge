#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Motor Predictive Maintenance System - Version 2.0
개선된 3단계 RUL(잔존 수명) 예측 시스템

Changes from v1:
- 3단계 건강 상태 분류 (NORMAL → WARNING → CRITICAL)
- IQR 기반 이상치 필터링
- 노이즈 필터링 개선 (slope 임계값)
- RUL 시간 계산 보정
- 추가 진단 정보 제공

Author: MOBY Team
Date: 2025-12-02
"""

import time
import signal
import threading
import json
import collections
import numpy as np
import RPi.GPIO as GPIO
import paho.mqtt.client as mqtt
from datetime import datetime, timedelta

# ==============================
# Config
# ==============================
AIN1 = 27
AIN2 = 22
PWMA = 18
STBY = 23

PWM_FREQ = 100     # Hz
DUTY = 68          # 0..100 (%)

stop_flag = False

# -----------------------------
# [PREDICTION CONFIG] 예지보전 설정 (3단계 기준)
# -----------------------------
# 베이스라인: 11/25일 초기 안정 구간 약 4583ms
# WARNING: 베이스라인 대비 +5% (약 4812ms)
# CRITICAL: 베이스라인 대비 +10% (약 5041ms) - 즉각 조치 필요
BASELINE_MS = 4583
WARNING_THRESHOLD_MS = int(BASELINE_MS * 1.05)   # 4812ms
FAILURE_THRESHOLD_MS = int(BASELINE_MS * 1.10)   # 5041ms

# 추세 분석을 위한 윈도우 크기 (최소 20개 이상 권장)
TREND_WINDOW_SIZE = 50
MIN_SAMPLES_FOR_PREDICTION = 20  # 예측 시작 최소 샘플 수

# slope 임계값: 노이즈 필터링 (사이클당 0.1ms 이상 증가 시에만 열화로 판단)
SLOPE_NOISE_THRESHOLD = 0.1       

# -----------------------------
# IR sensor + MQTT
# -----------------------------
IR_PIN = 17
DEAD_TIME_MS = 200
AVG_WINDOW = 10
PRINT_EVERY = 1
MQTT_BROKER = "192.168.80.143"
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
        # 3단계 상태 아이콘
        icon_map = {
            "NORMAL": "🟢",
            "WARNING": "🟡", 
            "CRITICAL": "🔴",
            "CALCULATING": "⚪"
        }
        icon = icon_map.get(status, "⚪")
        
        log_str = f"[MQTT] {icon} {status} | Cycle={msg['cycles']} | Avg={msg['avg_cycle_ms']}ms"
        
        # slope 정보 추가
        if msg.get('slope') is not None:
            slope_trend = "↑" if msg['slope'] > 0 else ("↓" if msg['slope'] < 0 else "→")
            log_str += f" | Trend: {slope_trend}{abs(msg['slope'])}ms/cycle"
        
        # RUL 정보 (유효한 경우만)
        if msg.get('rul_hours') is not None and msg.get('rul_cycles', -1) >= 0:
            log_str += f" | 🕒 RUL: {msg['rul_hours']}h ({msg['fail_time']})"
        elif msg.get('rul_cycles') == -1:
            log_str += " | ✅ Stable (no degradation)"
        
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
    현재 이동평균 값을 바탕으로 건강 상태 및 RUL을 예측합니다.
    
    Returns:
        health: "CALCULATING" | "NORMAL" | "WARNING" | "CRITICAL"
        rul_cycles: 예상 잔존 사이클 수 (None = 계산 불가, -1 = 무한/안정)
        slope: 사이클당 ms 증가율
    """
    # 데이터 부족 시 스킵
    if len(trend_buffer) < MIN_SAMPLES_FOR_PREDICTION:
        samples_needed = MIN_SAMPLES_FOR_PREDICTION - len(trend_buffer)
        return "CALCULATING", None, 0.0

    # 1. 상태 진단 (3단계)
    if current_avg_ms >= FAILURE_THRESHOLD_MS:
        health = "CRITICAL"  # 즉각 조치 필요
    elif current_avg_ms >= WARNING_THRESHOLD_MS:
        health = "WARNING"   # 주의 관찰 필요
    else:
        health = "NORMAL"    # 정상

    # 2. 기울기 계산 (Numpy Linear Regression)
    y = np.array(trend_buffer)
    x = np.arange(len(y))
    
    # 이상치 필터링: IQR 방식으로 극단값 제거
    q1, q3 = np.percentile(y, [25, 75])
    iqr = q3 - q1
    valid_mask = (y >= q1 - 1.5 * iqr) & (y <= q3 + 1.5 * iqr)
    
    if valid_mask.sum() < MIN_SAMPLES_FOR_PREDICTION // 2:
        # 유효 데이터가 너무 적으면 전체 사용
        slope, intercept = np.polyfit(x, y, 1)
    else:
        slope, intercept = np.polyfit(x[valid_mask], y[valid_mask], 1)

    # 3. 잔존 수명(RUL) 예측
    rul_cycles = None
    
    # slope가 노이즈 수준 이상으로 증가 중일 때만 RUL 계산
    if slope > SLOPE_NOISE_THRESHOLD:
        remaining_ms = FAILURE_THRESHOLD_MS - current_avg_ms
        if remaining_ms > 0:
            rul_cycles = int(remaining_ms / slope)
        else:
            rul_cycles = 0  # 이미 임계값 도달
    elif slope < -SLOPE_NOISE_THRESHOLD:
        # 개선 추세 (음의 기울기) - 상태 호전 중
        rul_cycles = -1  # 무한대/안정 표시
    else:
        # 안정 상태 (기울기 거의 0)
        rul_cycles = -1  # 현재 상태 유지 중

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
        
        # RUL이 유효하고 양수일 때만 시간 계산
        if rul_cycles is not None and rul_cycles > 0:
            # 미래 평균 사이클 시간 보정: (현재 + 임계값) / 2
            avg_future_ms = (avg_ms + FAILURE_THRESHOLD_MS) / 2.0
            seconds_left = rul_cycles * (avg_future_ms / 1000.0)
            rul_hours = round(seconds_left / 3600.0, 2)
            fail_dt = datetime.now() + timedelta(seconds=seconds_left)
            fail_time_str = fail_dt.strftime("%Y-%m-%d %H:%M:%S")
            
        msg = {
            "cycles": cycle_count,
            "last_cycle_ms": round(dt_ms, 2),
            "avg_cycle_ms": round(avg_ms, 2),
            "timestamp_ns": t_ns,
            "health": health_status,       # NORMAL / WARNING / CRITICAL
            "slope": slope,
            "rul_cycles": rul_cycles,
            "rul_hours": rul_hours,
            "fail_time": fail_time_str,
            # 추가 진단 정보
            "baseline_ms": BASELINE_MS,
            "warning_threshold_ms": WARNING_THRESHOLD_MS,
            "failure_threshold_ms": FAILURE_THRESHOLD_MS,
            "deviation_percent": round((avg_ms - BASELINE_MS) / BASELINE_MS * 100, 2)
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

    print(f"[SYSTEM] 3-Level RUL Prediction System Started.")
    print(f"[CONFIG] Baseline: {BASELINE_MS}ms")
    print(f"[CONFIG] Warning Threshold: {WARNING_THRESHOLD_MS}ms (+{round((WARNING_THRESHOLD_MS/BASELINE_MS-1)*100, 1)}%)")
    print(f"[CONFIG] Critical Threshold: {FAILURE_THRESHOLD_MS}ms (+{round((FAILURE_THRESHOLD_MS/BASELINE_MS-1)*100, 1)}%)")
    
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
