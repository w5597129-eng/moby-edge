#!/usr/bin/env python3
# Strict ASCII, no special characters

import time
import json
import signal

import board
import busio
from adafruit_ads1x15.ads1115 import ADS1115
from adafruit_ads1x15.analog_in import AnalogIn
import paho.mqtt.client as mqtt

# ==============================
# Configuration
# ==============================
DEVICE_ID = "Motor_A_01"
SENSOR_TYPE = "sound"
SENSOR_MODEL = "AnalogMic_AO"

# ADS1115
ADC_CHANNEL = 1        # A1 (sound sensor AO connected to A1)
ADC_GAIN = 1           # For 3.3V sensors, 1 or 2 is recommended

# MQTT
BROKER_HOST = "192.168.80.208"   # If broker runs on Pi, use localhost; otherwise, use the broker IP
BROKER_PORT = 1883
MQTT_TOPIC = "factory/sensor/sound"
MQTT_QOS = 0
MQTT_RETAIN = False

PUBLISH_INTERVAL_SEC = 0.5  # Publish interval in seconds

# ==============================
# Signal handler
# ==============================
stop_flag = False
def handle_sig(sig, frame):
    global stop_flag
    stop_flag = True
signal.signal(signal.SIGINT, handle_sig)
signal.signal(signal.SIGTERM, handle_sig)

# ==============================
# I2C / ADS1115 init
# ==============================
i2c = busio.I2C(board.SCL, board.SDA)
ads = ADS1115(i2c)
ads.gain = ADC_GAIN
ain = AnalogIn(ads, ADC_CHANNEL)  # A1

# ==============================
# MQTT connection
# ==============================
client = mqtt.Client(client_id="pi_sound_ads1115_pub")
client.connect(BROKER_HOST, BROKER_PORT, 60)
client.loop_start()

print("ADS1115 A1 Sound to MQTT Publisher")
print(f"Broker: {BROKER_HOST}:{BROKER_PORT}, Topic: {MQTT_TOPIC}")
print("Press Ctrl+C to stop.")

# ==============================
# Main loop
# ==============================
try:
    while not stop_flag:
        voltage = ain.voltage
        raw_value = ain.value  # 0..32767
        ts_ns = time.time_ns()

        payload = {
            "device_id": DEVICE_ID,
            "sensor_type": SENSOR_TYPE,
            "sensor_model": SENSOR_MODEL,
            "fields": {
                "sound_voltage": float(voltage),  # volts
                "sound_raw": int(raw_value)       # 0..32767
            },
            "timestamp_ns": int(ts_ns)
        }                      

        client.publish(MQTT_TOPIC, json.dumps(payload), qos=MQTT_QOS, retain=MQTT_RETAIN)
        print(f"PUB {MQTT_TOPIC} V={voltage:0.4f}V raw={raw_value:5d}", end="\r")

        time.sleep(PUBLISH_INTERVAL_SEC)

except Exception as e:
    print(f"\nERROR: {e}")

finally:
    client.loop_stop()
    client.disconnect()
    print("\nStopped.")
