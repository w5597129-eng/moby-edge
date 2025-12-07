#!/usr/bin/env python3
"""
MQTT to CSV Logger
Captures all sensor data from 'factory/#' topics and saves them to local CSV files.
Supports labeling for data collection scenarios.
"""

import os
import json
import time
import csv
import argparse
import sys
from datetime import datetime
from pathlib import Path
import paho.mqtt.client as mqtt

# Try to load .env
try:
    from dotenv import load_dotenv
    # Load from project root (assuming script is in src/)
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)
except ImportError:
    print("[WARN] python-dotenv not installed, utilizing system environment variables only.")

# ==============================
# Config
# ==============================
# Only from env or default fallback if missing
BROKER = os.getenv("MQTT_BROKER", "localhost") 
PORT = int(os.getenv("MQTT_PORT", 1883))
TOPIC_ROOT = "factory/#"
LOG_DIR = Path("data/logger")

# Map specific topics to friendly filenames
TOPIC_MAP = {
    "factory/sensor/dht11": "dht11",
    "factory/sensor/vibration": "vibration",
    "factory/sensor/sound": "sound",
    "factory/sensor/accel_gyro": "accel_gyro",
    "factory/sensor/pressure": "pressure",
    "factory/conveyor/ir": "conveyor_ir"
}

file_handles = {}
csv_writers = {}
CURRENT_LABEL = "NORMAL"  # Default label

def ensure_log_dir():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

def flatten_payload(topic, payload):
    flat_data = {}
    
    # 1. Timestamp
    ts_ns = payload.get("timestamp_ns")
    if ts_ns:
        flat_data["timestamp_ns"] = ts_ns
        try:
            dt = datetime.fromtimestamp(ts_ns / 1e9)
            flat_data["time"] = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        except:
            flat_data["time"] = get_timestamp()
    else:
        flat_data["time"] = get_timestamp()
        flat_data["timestamp_ns"] = int(time.time() * 1e9)

    # 2. Extract Data Fields
    if "fields" in payload and isinstance(payload["fields"], dict):
        flat_data.update(payload["fields"])
    else:
        exclude = ["timestamp_ns", "sensor_type", "sensor_model"]
        for k, v in payload.items():
            if k not in exclude:
                flat_data[k] = v
    
    # 3. Add Label
    flat_data["label"] = CURRENT_LABEL
                
    return flat_data

def write_to_csv(filename, data):
    filepath = LOG_DIR / f"{filename}.csv"
    file_exists = filepath.exists()
    
    if filename not in csv_writers:
        mode = "a" if file_exists else "w"
        f = open(filepath, mode, newline="", encoding="utf-8")
        file_handles[filename] = f
        
        # Ensure 'label' is in fieldnames, put it at the end or beginning
        base_keys = [k for k in data.keys() if k not in ["time", "timestamp_ns", "label"]]
        fieldnames = ["time", "timestamp_ns"] + sorted(base_keys) + ["label"]
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
            print(f"[LOG] Created new log file: {filepath}")
        
        csv_writers[filename] = (writer, fieldnames)
    
    writer, fieldnames = csv_writers[filename]
    
    # Filter to known fields
    row = {k: v for k, v in data.items() if k in fieldnames}
    
    try:
        writer.writerow(row)
        file_handles[filename].flush()
    except ValueError:
        pass

def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT Broker ({BROKER}) with result code {rc}")
    client.subscribe(TOPIC_ROOT)
    print(f"Subscribed to {TOPIC_ROOT}")

def on_message(client, userdata, msg):
    try:
        topic = msg.topic
        payload_str = msg.payload.decode("utf-8")
        payload = json.loads(payload_str)
        
        filename = TOPIC_MAP.get(topic)
        if not filename:
             filename = topic.split("/")[-1]
        
        clean_data = flatten_payload(topic, payload)
        write_to_csv(filename, clean_data)
        
        print(f"\r[DATA] {filename}: {len(clean_data)} fields | Label: {CURRENT_LABEL}", end="")
        
    except Exception as e:
        print(f"\n[ERR] {e}")

def main():
    global CURRENT_LABEL
    ensure_log_dir()
    
    # Parse args for label
    parser = argparse.ArgumentParser(description="MQTT Data Logger with Labeling")
    parser.add_argument("--label", "-l", type=str, help="Label for this data collection session (e.g. NORMAL, FAULT_1)")
    # Removed --broker arg to enforce .env usage as requested, or keep it as override? 
    # User said "only use the one specified in .env". I'll remove the arg to be strict, or keep it as optional for debugging.
    # The user instruction was "make it use only environment variable". 
    # I will allow --broker for flexibility but default is ENV.
    parser.add_argument("--broker", "-b", type=str, default=BROKER, help="MQTT Broker IP (default: from .env)")
    args = parser.parse_args()
    
    if args.label:
        CURRENT_LABEL = args.label
    else:
        # Interactive input if not provided
        print("=== MQTT to CSV Data Logger ===")
        print(f"[INIT] Loaded Broker from .env: {BROKER}")
        print("Enter a label for this session (default: NORMAL): ", end="")
        user_input = input().strip()
        if user_input:
            CURRENT_LABEL = user_input
            
    broker_addr = args.broker

    print(f"\n[CONFIG] Target: {LOG_DIR.absolute()}")
    print(f"[CONFIG] Label : {CURRENT_LABEL}")
    print(f"[CONFIG] Broker: {broker_addr}")
    print("Press Ctrl+C to stop.\n")
    
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    
    try:
        client.connect(broker_addr, PORT, 60)
        client.loop_forever()
    except KeyboardInterrupt:
        print("\nStopping logger...")
    finally:
        for f in file_handles.values():
            try: f.close()
            except: pass
        print("Bye.")

if __name__ == "__main__":
    main()
