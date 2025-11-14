#!/usr/bin/env python3
"""
Simple MQTT publisher+subscriber to verify broker and topics work.
Publishes a few test messages to `factory/edge/raw` and `factory/edge/anomaly`
and subscribes to the same topics to confirm delivery.

Usage: python scripts/mqtt_test_pub.py
"""

import time
import json
import threading

import paho.mqtt.client as mqtt

BROKER = "localhost"
RAW_TOPIC = "factory/edge/raw"
ANOM_TOPIC = "factory/edge/anomaly"

received = []


def on_connect(client, userdata, flags, rc):
    print("[mqtt_test] Connected to broker, rc=", rc)
    client.subscribe(RAW_TOPIC)
    client.subscribe(ANOM_TOPIC)


def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode('utf-8')
    except Exception:
        payload = str(msg.payload)
    print("[mqtt_test] RECV", msg.topic, payload)
    received.append((msg.topic, payload))


def run_test(count=3, delay=0.5):
    client = mqtt.Client(client_id="mqtt_test_pub")
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(BROKER, 1883, 60)
    client.loop_start()

    # allow connect
    time.sleep(1.0)

    print("[mqtt_test] Publishing test messages...")
    for i in range(count):
        raw = {"test": "raw", "i": i, "ts": int(time.time() * 1e9)}
        anom = {"test": "anom", "i": i, "score": 0.1 * i, "ts": int(time.time() * 1e9)}
        client.publish(RAW_TOPIC, json.dumps(raw))
        client.publish(ANOM_TOPIC, json.dumps(anom))
        time.sleep(delay)

    # wait for messages to be received by subscriber callback
    time.sleep(1.5)
    client.loop_stop()
    client.disconnect()

    print(f"[mqtt_test] Done. Received {len(received)} messages.")
    for t, p in received:
        print("  -", t, p)


if __name__ == '__main__':
    run_test()
