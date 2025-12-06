
import os
import sys
import json
import time
import csv
import argparse
from datetime import datetime
import paho.mqtt.client as mqtt

# Default Configuration
DEFAULT_BROKER = "192.168.80.208" # Based on sensor_final.py
DEFAULT_PORT = 1883
TOPIC_IMU = "factory/sensor/accel_gyro"

class DataCollector:
    def __init__(self, broker, port, label, duration, output_dir="data/training_raw"):
        self.broker = broker
        self.port = port
        self.label = label
        self.duration = duration
        self.output_dir = output_dir
        self.client = mqtt.Client("remote_collector_" + datetime.now().strftime("%H%M%S"))
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        
        self.data_buffer = []
        self.start_time = None
        self.is_collecting = False
        self.sample_count = 0
        
        os.makedirs(self.output_dir, exist_ok=True)
        
    def on_connect(self, client, userdata, flags, rc):
        print(f"Connected to Broker {self.broker}:{self.port} (RC={rc})")
        client.subscribe(TOPIC_IMU)
        print(f"Subscribed to {TOPIC_IMU}")
        print("Waiting for data stream...")

    def on_message(self, client, userdata, msg):
        if not self.is_collecting:
            return

        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            # Expecting payload from sensor_final.py (100Hz)
            # { "fields": { "accel_x": ... }, "timestamp_ns": ... }
            
            fields = payload.get("fields", {})
            ts = payload.get("timestamp_ns", 0)
            
            # Simple validation
            if "accel_x" not in fields:
                return
                
            row = [
                ts,
                fields.get("accel_x", 0),
                fields.get("accel_y", 0),
                fields.get("accel_z", 0),
                fields.get("gyro_x", 0),
                fields.get("gyro_y", 0),
                fields.get("gyro_z", 0)
            ]
            self.data_buffer.append(row)
            self.sample_count += 1
            
            # Progress indicator (every 100 samples ~ 1 sec)
            if self.sample_count % 100 == 0:
                elapsed = time.time() - self.start_time
                print(f"Collecting... {elapsed:.1f}/{self.duration}s ({self.sample_count} samples)", end='\r')
                
        except Exception as e:
            print(f"Error parsing message: {e}")

    def start(self):
        try:
            print(f"Connecting to {self.broker}...")
            self.client.connect(self.broker, self.port, 60)
            self.client.loop_start()
            
            # Wait for connection
            time.sleep(1)
            
            print(f"\n[START] Collecting data for Label: '{self.label}'")
            print(f"Target Duration: {self.duration} seconds")
            
            self.is_collecting = True
            self.start_time = time.time()
            
            while (time.time() - self.start_time) < self.duration:
                time.sleep(0.1)
                
            self.is_collecting = False
            self.client.loop_stop()
            self.client.disconnect()
            
            self.save_csv()
            
        except KeyboardInterrupt:
            self.is_collecting = False
            self.save_csv()
            print("\nCollection interrupted by user.")
        except Exception as e:
            print(f"\nError: {e}")

    def save_csv(self):
        if not self.data_buffer:
            print("\nNo data collected.")
            return

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.label}_{timestamp_str}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        print(f"\nSaving {len(self.data_buffer)} samples to {filepath}...")
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Header
            writer.writerow(['timestamp_ns', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z'])
            writer.writerows(self.data_buffer)
            
        print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect sensor data from MQTT for training")
    parser.add_argument("--label", required=True, help="Class label (e.g., normal, yellow, red)")
    parser.add_argument("--duration", type=int, default=60, help="Collection duration in seconds")
    parser.add_argument("--broker", default=DEFAULT_BROKER, help="MQTT Broker IP")
    
    args = parser.parse_args()
    
    collector = DataCollector(args.broker, DEFAULT_PORT, args.label, args.duration)
    collector.start()
