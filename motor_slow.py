#!/usr/bin/env python3
# -*- coding: us-ascii -*-

import time
import signal
import threading
import json
import RPi.GPIO as GPIO
import paho.mqtt.client as mqtt

# ======================================
# Config
# ======================================
AIN1 = 27
AIN2 = 22
PWMA = 18
STBY = 23

PWM_FREQ = 100          # Hz

# One conveyor cycle (average, seconds)
CYCLE_SEC = 3.63

# Duty settings
NORMAL_DUTY = 75        # normal speed duty
DIP_DUTY = 30           # slowed-down duty (do not use 0)  # red_30 yellow_60

# Where in the cycle the dip happens (0.0 ~ 1.0)
# Example: dip starts at 40% of the cycle and lasts 20% of the cycle
DIP_START_RATIO = 0.40
DIP_DURATION_RATIO = 0.20

LOOP_SLEEP_SEC = 0.02   # main loop sleep

stop_flag = False

# ======================================
# SIGINT handler
# ======================================
def handle_sigint(sig, frame):
    global stop_flag
    print("\n[MOTOR] Interrupt received. Stopping...")
    stop_flag = True

signal.signal(signal.SIGINT, handle_sigint)

# ======================================
# GPIO init (BCM mode)
# ======================================
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

# -----------------------------
# IR sensor + MQTT (merged from IR_mqtt_pub.py)
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

def now_ns():
    return time.monotonic_ns()

def init_mqtt():
    global mqtt_client
    try:
        mqtt_client = mqtt.Client(client_id=MQTT_CLIENT_ID)
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
    except Exception:
        mqtt_client = None

def _publish_ir(msg: dict):
    # Log the outgoing MQTT payload to the terminal so users can see publishes
    try:
        print(f"[MQTT PUBLISH] topic={MQTT_TOPIC} payload={json.dumps(msg)}")
    except Exception:
        print("[MQTT PUBLISH] (could not serialize msg)")

    if mqtt_client:
        try:
            mqtt_client.publish(MQTT_TOPIC, json.dumps(msg))
        except Exception:
            pass

def record_hit(t_ns):
    global last_hit_ns, dead_until_ns, cycle_count, cycle_times_ms
    if t_ns < dead_until_ns:
        return
    if last_hit_ns is None:
        last_hit_ns = t_ns
        dead_until_ns = t_ns + DEAD_TIME_MS * 1_000_000
        return
    dt_ms = (t_ns - last_hit_ns) / 1_000_000.0
    last_hit_ns = t_ns
    dead_until_ns = t_ns + DEAD_TIME_MS * 1_000_000
    if dt_ms < DEAD_TIME_MS * 1.2:
        return
    cycle_count += 1
    cycle_times_ms.append(dt_ms)
    if len(cycle_times_ms) > AVG_WINDOW:
        cycle_times_ms = cycle_times_ms[-AVG_WINDOW:]
    if cycle_count % PRINT_EVERY == 0:
        avg_ms = sum(cycle_times_ms) / len(cycle_times_ms) if cycle_times_ms else float('nan')
        msg = {
            "cycles": cycle_count,
            "last_cycle_ms": round(dt_ms, 2),
            "avg_cycle_ms": round(avg_ms, 2) if avg_ms == avg_ms else None
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

# ======================================
# Main
# ======================================
def main():
    init_gpio_bcm()
    # Start IR MQTT publisher thread
    init_mqtt()
    start_ir_thread()
    pwm = GPIO.PWM(PWMA, PWM_FREQ)
    pwm.start(0)

    print("[MOTOR] Cycle-based step slowdown. Ctrl+C to stop.")
    GPIO.output(STBY, GPIO.HIGH)

    # fixed forward direction
    GPIO.output(AIN1, GPIO.HIGH)
    GPIO.output(AIN2, GPIO.LOW)

    # precompute dip window in seconds
    dip_start_sec = CYCLE_SEC * DIP_START_RATIO
    dip_end_sec = dip_start_sec + CYCLE_SEC * DIP_DURATION_RATIO

    # start at normal duty
    current_duty = NORMAL_DUTY
    pwm.ChangeDutyCycle(current_duty)
    print(f"[MOTOR] duty={current_duty}% (normal start)")

    cycle_t0 = time.time()

    try:
        while not stop_flag:
            now = time.time()
            elapsed = now - cycle_t0

            # wrap elapsed into [0, CYCLE_SEC)
            phase = elapsed % CYCLE_SEC

            # decide duty based on phase position in the cycle
            if dip_start_sec <= phase < dip_end_sec:
                target_duty = DIP_DUTY
            else:
                target_duty = NORMAL_DUTY

            if target_duty != current_duty:
                current_duty = target_duty
                pwm.ChangeDutyCycle(current_duty)
                if current_duty == DIP_DUTY:
                    print(f"[MOTOR] duty={current_duty}% (dip)")
                else:
                    print(f"[MOTOR] duty={current_duty}% (normal)")

            time.sleep(LOOP_SLEEP_SEC)

    finally:
        pwm.ChangeDutyCycle(0)
        pwm.stop()
        GPIO.output(AIN1, GPIO.LOW)
        GPIO.output(AIN2, GPIO.LOW)
        GPIO.output(STBY, GPIO.LOW)
        GPIO.cleanup()
        # stop MQTT
        try:
            if mqtt_client:
                mqtt_client.loop_stop()
                mqtt_client.disconnect()
        except Exception:
            pass
        print("[MOTOR] Clean exit.")

if __name__ == "__main__":
    main()
