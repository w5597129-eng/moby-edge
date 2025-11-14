#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import json
import threading
import signal

import adafruit_dht
import board
import busio
from adafruit_ads1x15.ads1115 import ADS1115
from adafruit_ads1x15.analog_in import AnalogIn

import paho.mqtt.client as mqtt
import RPi.GPIO as GPIO

# ==============================
# Config
# ==============================
BROKER = "localhost"
PORT = 1883
TOPIC_DHT = "factory/sensor/dht11"
TOPIC_VIB = "factory/sensor/vibration"

PUBLISH_INTERVAL_SEC_DHT = 1

# ADS1115 (I2C)
ADS_ADDR = 0x48          # 보통 0x48
ADS_GAIN = 1             # +-4.096V 범위(센서 전압에 맞게 조정)
ADS_CHANNEL = 0          # A0

# 진동 출력 주기 제어
READ_HZ_VIB = 200        # 내부 읽기 주기(버퍼링용)
PRINT_EVERY_SEC = 1  # 콘솔에는 이 간격으로 한 줄만 출력

# TB6612FNG (BCM numbering)
AIN1 = 27      # TB6612 AIN1
AIN2 = 22      # TB6612 AIN2
PWMA = 18      # TB6612 PWMA (PWM-capable)
STBY = 23      # TB6612 STBY (must be HIGH)

PWM_FREQ = 100
DUTY = 80
RUN_SEC = 5
STOP_SEC = 2

stop_event = threading.Event()
pwm = None

# ==============================
# GPIO init (robust)
# ==============================
def init_gpio_bcm():
    """
    Make BCM mode robust even if a previous run left BOARD mode set.
    """
    GPIO.setwarnings(False)
    mode = GPIO.getmode()  # None, GPIO.BOARD(10), or GPIO.BCM(11)
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
# ADS1115 init
# ==============================
def init_ads1115():
    """
    Initialize ADS1115 and return (ads, channel)
    """
    i2c = busio.I2C(board.SCL, board.SDA)
    ads = ADS1115(i2c, address=ADS_ADDR)
    ads.gain = ADS_GAIN
    ch = AnalogIn(ads, ADS_CHANNEL)
    return ads, ch

# ==============================
# Helpers
# ==============================
def _wait_with_stop(seconds):
    end = time.time() + seconds
    while time.time() < end and not stop_event.is_set():
        time.sleep(0.01)

def _now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S")

# ==============================
# Threads
# ==============================
def dht11_publisher(dht_device, mqtt_client):
    print("[DHT] Publisher started.")
    while not stop_event.is_set():
        try:
            temp = dht_device.temperature
            hum = dht_device.humidity
            if temp is not None and hum is not None:
                payload = {
                    "temperature": temp,
                    "humidity": hum,
                    "timestamp": _now_str()
                }
                mqtt_client.publish(TOPIC_DHT, json.dumps(payload))
                print(f"[DHT] {payload}")
            else:
                print("[DHT] Sensor read failed. Retrying...")
        except RuntimeError as e:
            print("[DHT] RuntimeError:", e.args[0])
        except Exception as e:
            print("[DHT] Unexpected error:", e)
            break
        time.sleep(PUBLISH_INTERVAL_SEC_DHT)
    print("[DHT] Publisher stopping...")

def vibration_reader(ch, mqtt_client):
    """
    Read ADS1115 channel continuously, print one line every PRINT_EVERY_SEC,
    and publish MQTT each print.
    """
    print("[VIB] Reader started.")
    read_interval = 1.0 / max(1, READ_HZ_VIB)
    last_print = 0.0
    accum_raw = 0
    accum_volt = 0.0
    n = 0

    while not stop_event.is_set():
        try:
            raw = ch.value          # 0..32767
            volt = ch.voltage       # Volts
        except Exception as e:
            # I2C 에러 등
            print("[VIB] Read error:", e)
            time.sleep(1)
            continue

        # 간단한 평균으로 노이즈 완화
        accum_raw += raw
        accum_volt += volt
        n += 1

        # 콘솔·MQTT 출력은 과도하지 않게 조절
        now = time.time()
        if now - last_print >= PRINT_EVERY_SEC:
            avg_raw = int(accum_raw / max(1, n))
            avg_volt = accum_volt / max(1, n)

            payload = {
                "adc_raw": avg_raw,
                "voltage": round(avg_volt, 6),
                "timestamp": _now_str()
            }
            mqtt_client.publish(TOPIC_VIB, json.dumps(payload))
            print(f"[VIB] {payload}")

            # 리셋
            accum_raw = 0
            accum_volt = 0.0
            n = 0
            last_print = now

        time.sleep(read_interval)

    print("[VIB] Reader stopping...")

def motor_runner():
    print("[MOTOR] Runner started.")
    try:
        GPIO.output(STBY, GPIO.HIGH)  # enable driver
        while not stop_event.is_set():
            GPIO.output(AIN1, GPIO.HIGH)
            GPIO.output(AIN2, GPIO.LOW)
            pwm.ChangeDutyCycle(DUTY)
            print("[MOTOR] Forward...")
            _wait_with_stop(RUN_SEC)

            pwm.ChangeDutyCycle(0)
            GPIO.output(AIN1, GPIO.LOW)
            GPIO.output(AIN2, GPIO.LOW)
            print("[MOTOR] Stop.")
            _wait_with_stop(STOP_SEC)
    finally:
        print("[MOTOR] Runner stopping...")

# ==============================
# Graceful shutdown
# ==============================
def handle_sigint(sig, frame):
    # pylint: disable=unused-argument
    print("\n[MAIN] Caught interrupt. Shutting down...")
    stop_event.set()

signal.signal(signal.SIGINT, handle_sigint)

# ==============================
# Main
# ==============================
if __name__ == "__main__":
    # Robust GPIO init (fixes "A different mode has already been set!")
    init_gpio_bcm()

    # PWM
    pwm = GPIO.PWM(PWMA, PWM_FREQ)
    pwm.start(0)

    # DHT11
    dhtDevice = adafruit_dht.DHT11(board.D4, use_pulseio=False)  # BCM 4

    # ADS1115
    ads, vib_ch = init_ads1115()

    # MQTT
    client = mqtt.Client("pi_pub_edge")
    client.connect(BROKER, PORT, 60)
    client.loop_start()

    # Start threads
    t_dht = threading.Thread(target=dht11_publisher, args=(dhtDevice, client), daemon=True)
    t_vib = threading.Thread(target=vibration_reader, args=(vib_ch, client), daemon=True)
    t_motor = threading.Thread(target=motor_runner, daemon=True)
    t_dht.start()
    t_vib.start()
    t_motor.start()

    print("[MAIN] Running. Press Ctrl+C to stop.")
    try:
        while not stop_event.is_set():
            time.sleep(0.2)
    finally:
        # Cleanup
        try:
            pwm.ChangeDutyCycle(0)
            pwm.stop()
        except Exception:
            pass

        GPIO.output(AIN1, GPIO.LOW)
        GPIO.output(AIN2, GPIO.LOW)
        GPIO.output(STBY, GPIO.LOW)
        GPIO.cleanup()

        try:
            dhtDevice.exit()
        except Exception:
            pass

        try:
            client.loop_stop()
            client.disconnect()
        except Exception:
            pass

        print("[MAIN] Clean exit.")
