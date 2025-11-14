#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Sensor Publisher for MOBY Edge Node
- DHT11 (GPIO)
- SEN0209 vibration (ADS1115 A0)
- SZH-EK087 sound (ADS1115 A1)
- MPU-6050 accel/gyro (I2C)
- BMP085/BMP180 pressure (I2C)

Always shows 5 fixed lines in terminal (ASCII-only).
Printed values == published values (same rounding).
"""

import sys
try:
    # Avoid UnicodeEncodeError in Thonny
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import time, json, signal
import adafruit_dht, board, busio
from adafruit_ads1x15.ads1115 import ADS1115
from adafruit_ads1x15.analog_in import AnalogIn
#!/usr/bin/env python3
import time, json
import numpy as np
import paho.mqtt.client as mqtt
import joblib
from feature_extractor import extract_features

# ==============================
# MODEL LOAD
# ==============================
MODEL_PATH = "/home/wise/python/models/isolation_forest.pkl"
SCALER_PATH = "/home/wise/python/models/scaler_if.pkl"
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ==============================
# MQTT SETUP
# ==============================
BROKER = "localhost"
RAW_TOPIC = "factory/edge/raw"
ANOM_TOPIC = "factory/edge/anomaly"
client = mqtt.Client()
client.connect(BROKER, 1883, 60)

# ==============================
# WINDOW BUFFER
# ==============================
WINDOW_SIZE = 200
buffer = []  # holds recent accel magnitudes


def read_all_sensors():
    """Initialize sensors on first call and return a dict of readings.
    Missing sensors return None for their values.
    """
    if not hasattr(read_all_sensors, "inited"):
        read_all_sensors.inited = True
        # lazy imports / initialization to keep top imports minimal
        try:
            import board, busio
            from adafruit_ads1x15.ads1115 import ADS1115
            from adafruit_ads1x15.analog_in import AnalogIn
            import adafruit_dht
            import smbus2
            read_all_sensors.board = board
            read_all_sensors.busio = busio
            read_all_sensors.ADS1115 = ADS1115
            read_all_sensors.AnalogIn = AnalogIn
            read_all_sensors.adafruit_dht = adafruit_dht
            read_all_sensors.smbus2 = smbus2
        except Exception:
            # hardware libs may be unavailable; mark as None
            read_all_sensors.board = None
            read_all_sensors.busio = None
            read_all_sensors.ADS1115 = None
            read_all_sensors.AnalogIn = None
            read_all_sensors.adafruit_dht = None
            read_all_sensors.smbus2 = None

        # try init sensors
        read_all_sensors.dht = None
        read_all_sensors.ads = None
        read_all_sensors.analog_vib = None
        read_all_sensors.analog_sound = None
        read_all_sensors.mpu_bus = None
        read_all_sensors.mpu_addr = None
        read_all_sensors.bmp = None

        try:
            if read_all_sensors.adafruit_dht and read_all_sensors.board:
                read_all_sensors.dht = read_all_sensors.adafruit_dht.DHT11(read_all_sensors.board.D4, use_pulseio=False)
        except Exception:
            read_all_sensors.dht = None

        try:
            if read_all_sensors.busio and read_all_sensors.ADS1115:
                i2c = read_all_sensors.busio.I2C(read_all_sensors.board.SCL, read_all_sensors.board.SDA)
                ads = read_all_sensors.ADS1115(i2c)
                read_all_sensors.ads = ads
                read_all_sensors.analog_vib = read_all_sensors.AnalogIn(ads, 0)
                read_all_sensors.analog_sound = read_all_sensors.AnalogIn(ads, 1)
        except Exception:
            read_all_sensors.ads = None

        try:
            if read_all_sensors.smbus2:
                bus = read_all_sensors.smbus2.SMBus(1)
                read_all_sensors.mpu_bus = bus
                # detect MPU6050
                for a in (0x68, 0x69):
                    try:
                        bus.read_byte_data(a, 0x75)
                        read_all_sensors.mpu_addr = a
                        bus.write_byte_data(a, 0x6B, 0x00)
                        break
                    except Exception:
                        continue
        except Exception:
            read_all_sensors.mpu_bus = None

        try:
            # BMP optional
            import Adafruit_BMP.BMP085 as BMP085
            read_all_sensors.bmp = BMP085.BMP085()
        except Exception:
            read_all_sensors.bmp = None

    # Read sensors
    out = {
        "accel_x": None,
        "accel_y": None,
        "accel_z": None,
        "gyro_x": None,
        "gyro_y": None,
        "gyro_z": None,
        "humidity": None,
        "temperature": None,
        "vibration": None,
        "pressure": None,
    }

    # MPU readings
    try:
        bus = read_all_sensors.mpu_bus
        addr = read_all_sensors.mpu_addr
        if bus and addr:
            def read_word_2c(bus, addr, reg):
                hi = bus.read_byte_data(addr, reg)
                lo = bus.read_byte_data(addr, reg + 1)
                val = (hi << 8) | lo
                if val >= 0x8000:
                    val = -((65535 - val) + 1)
                return val

            ACCEL_SENS, GYRO_SENS = 16384.0, 131.0
            ax = read_word_2c(bus, addr, 0x3B) / ACCEL_SENS
            ay = read_word_2c(bus, addr, 0x3D) / ACCEL_SENS
            az = read_word_2c(bus, addr, 0x3F) / ACCEL_SENS
            gx = read_word_2c(bus, addr, 0x43) / GYRO_SENS
            gy = read_word_2c(bus, addr, 0x45) / GYRO_SENS
            gz = read_word_2c(bus, addr, 0x47) / GYRO_SENS
            out.update({"accel_x": ax, "accel_y": ay, "accel_z": az, "gyro_x": gx, "gyro_y": gy, "gyro_z": gz})
    except Exception:
        pass

    # DHT11
    try:
        dht = read_all_sensors.dht
        if dht:
            # Only read humidity from DHT11 here. Temperature will come from BMP sensor.
            h = dht.humidity
            out["humidity"] = float(h) if h is not None else None
    except Exception:
        pass

    # ADS1115 analogs (vibration)
    try:
        vib = read_all_sensors.analog_vib
        if vib:
            out["vibration"] = float(vib.voltage)
    except Exception:
        pass

    # BMP
    try:
        bmp = read_all_sensors.bmp
        if bmp:
            # Read pressure and temperature from BMP sensor. Temperature is sourced
            # from BMP only (per requirement), so overwrite any other temp.
            out["pressure"] = float(bmp.read_pressure())/100.0
            try:
                out["temperature"] = float(bmp.read_temperature())
            except Exception:
                # Some BMP drivers may not implement read_temperature(); ignore if missing
                pass
    except Exception:
        pass

    return out


def main():
    client.loop_start()
    sampling_rate = 100.0  # Hz estimated (sleep 0.01)
    try:
        while True:
            data = read_all_sensors()
            # Publish raw JSON
            try:
                client.publish(RAW_TOPIC, json.dumps(data))
            except Exception:
                pass

            # append accel magnitude to buffer
            ax = data.get("accel_x")
            ay = data.get("accel_y")
            az = data.get("accel_z")
            if ax is not None and ay is not None and az is not None:
                mag = float(np.sqrt(ax*ax + ay*ay + az*az))
                buffer.append(mag)
                # keep buffer length bounded
                if len(buffer) > WINDOW_SIZE:
                    buffer.pop(0)

            # Run inference when buffer full
            if len(buffer) >= WINDOW_SIZE:
                window = np.array(buffer[-WINDOW_SIZE:])
                feats = extract_features(window, sampling_rate)
                X = np.array(feats).reshape(1, -1)
                try:
                    Xs = scaler.transform(X)
                except Exception:
                    Xs = X
                try:
                    score = float(model.decision_function(Xs)[0]) if hasattr(model, 'decision_function') else None
                except Exception:
                    score = None
                try:
                    label = int(model.predict(Xs)[0])
                except Exception:
                    label = None

                an = {"anomaly_score": score, "anomaly_label": label}
                payload = {"fields": an, "timestamp": int(time.time()*1e9)}
                try:
                    client.publish(ANOM_TOPIC, json.dumps(payload))
                except Exception:
                    pass

            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            client.loop_stop()
            client.disconnect()
        except Exception:
            pass


if __name__ == '__main__':
    main()
