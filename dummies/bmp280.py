#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BMP180 (using Adafruit BMP280 driver) - Pressure & Temperature Publisher
- Compatible with BMP180/BMP280
- I2C shared bus (SDA/SCL)
- MQTT publish
"""

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import time
import json
import signal
import paho.mqtt.client as mqtt
import board
import adafruit_bmp280

# ==============================
# Configuration
# ==============================
DEVICE_ID = "Env_A_01"
SENSOR_TYPE = "pressure"
SENSOR_MODEL = "BMP180"

BROKER_HOST = "192.168.80.208"
BROKER_PORT = 1883
MQTT_TOPIC = "factory/sensor/pressure"
PUBLISH_INTERVAL_SEC = 2.0

# ==============================
# Setup
# ==============================
i2c = board.I2C()  # SDA=BCM 2 (pin 3), SCL=BCM 3 (pin 5)
bmp = adafruit_bmp280.Adafruit_BMP280_I2C(i2c, address=0x77)  # BMP180 address 0x77
bmp.sea_level_pressure = 1013.25  # hPa

client = mqtt.Client("pi_bmp180_pub")
client.connect(BROKER_HOST, BROKER_PORT, 60)
client.loop_start()

stop_flag = False
def handle_sigint(sig, frame):
    global stop_flag
    stop_flag = True
signal.signal(signal.SIGINT, handle_sigint)

print("BMP180 Publisher started.")

# ==============================
# Main loop
# ==============================
while not stop_flag:
    try:
        temperature = bmp.temperature       # deg C
        pressure = bmp.pressure             # hPa
        altitude = bmp.altitude             # meters (optional)

        payload = {
            "device_id": DEVICE_ID,
            "sensor_type": SENSOR_TYPE,
            "sensor_model": SENSOR_MODEL,
            "temperature_c": round(temperature, 2),
            "pressure_hpa": round(pressure, 2),
            "altitude_m": round(altitude, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # ensure_ascii=True guarantees ASCII-only output
        client.publish(MQTT_TOPIC, json.dumps(payload, ensure_ascii=True))
        print("Published:", json.dumps(payload, ensure_ascii=True))

    except Exception as e:
        print("[BMP180] Read error:", repr(e))

    time.sleep(PUBLISH_INTERVAL_SEC)

client.loop_stop()
print("BMP180 Publisher stopped.")
