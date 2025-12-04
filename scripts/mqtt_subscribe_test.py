#!/usr/bin/env python3
"""
Simple MQTT subscriber to validate `sensor_final` payloads.
Usage:
  python scripts/mqtt_subscribe_test.py --duration 30
"""
import os
import time
import json
import argparse
import paho.mqtt.client as mqtt

DEFAULT_BROKER = os.getenv("MQTT_BROKER", "192.168.80.208")
DEFAULT_PORT = 1883

BASE_TOPICS = [
    "factory/sensor/dht11",
    "factory/sensor/vibration",
    "factory/sensor/sound",
    "factory/sensor/accel_gyro",
    "factory/sensor/pressure",
]
# also subscribe to inference topics
TOPICS = BASE_TOPICS + [t + "/inference" for t in BASE_TOPICS]

args_parser = argparse.ArgumentParser(description="Subscribe to sensor_final topics and print/validate messages")
args_parser.add_argument("--broker", default=DEFAULT_BROKER)
args_parser.add_argument("--port", default=DEFAULT_PORT, type=int)
args_parser.add_argument("--duration", default=0, type=int, help="Run duration in seconds (0 = run until Ctrl+C)")
args = args_parser.parse_args()

start_ts = time.time()


def now_hms():
    return time.strftime("%H:%M:%S")


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"[MQTT] Connected to {args.broker}:{args.port}")
        for t in TOPICS:
            client.subscribe(t)
            print(f"[MQTT] Subscribed to: {t}")
    else:
        print(f"[MQTT] Connect failed: rc={rc}")


def safe_load_json(payload):
    try:
        return json.loads(payload)
    except Exception as e:
        return None


def on_message(client, userdata, msg):
    s = msg.payload.decode(errors='replace')
    obj = safe_load_json(s)
    header = f"{now_hms()} | {msg.topic}"
    if obj is None:
        print(header + " | INVALID JSON ->", s)
        return
    # basic validation
    sensor_type = obj.get('sensor_type')
    ts = obj.get('timestamp_ns')
    fields = obj.get('fields')
    print(header + f" | sensor_type={sensor_type} timestamp_ns={ts}")
    if isinstance(fields, dict):
        # print up to 6 keys
        keys = list(fields.keys())
        sample = ", ".join(f"{k}={fields[k]}" for k in keys[:6])
        print("  fields:", sample)
    else:
        print("  fields: (missing or not a dict)")


client = mqtt.Client(client_id="sensor_sub_test")
client.on_connect = on_connect
client.on_message = on_message

try:
    client.connect(args.broker, args.port, 60)
except Exception as e:
    print("Could not connect to MQTT broker:", e)
    raise SystemExit(1)

client.loop_start()

try:
    if args.duration and args.duration > 0:
        # run for the requested duration
        end = time.time() + args.duration
        while time.time() < end:
            time.sleep(0.1)
    else:
        # run until Ctrl+C
        print("Running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1.0)
except KeyboardInterrupt:
    pass
finally:
    client.loop_stop()
    client.disconnect()
    print("Stopped subscriber.")
