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
import smbus2
import paho.mqtt.client as mqtt

# 버퍼 저장/재전송/정리용
import os, glob
import numpy as np

# Pressure sensor (BMP085/BMP180)
try:
    import Adafruit_BMP.BMP085 as BMP085  # pip3 install Adafruit-BMP
    HAS_BMP = True
except Exception:
    HAS_BMP = False

# ==============================
# Config
# ==============================
BROKER = "localhost"
PORT = 1883
# 버퍼 경로
BUFFER_DIR = "/home/wise/deployment/data/buffer/"
BUFFER_MAX_FILES = 100

def ensure_buffer_dir():
    os.makedirs(BUFFER_DIR, exist_ok=True)

def save_to_buffer(sensor_type, payload):
    ensure_buffer_dir()
    ts = payload.get("timestamp_ns", now_ns())
    fname = f"{sensor_type}_{ts}.npy"
    fpath = os.path.join(BUFFER_DIR, fname)
    # numpy로 저장: object array (payload dict)
    np.save(fpath, np.array([payload], dtype=object))

def resend_buffered(client):
    ensure_buffer_dir()
    npy_files = sorted(glob.glob(os.path.join(BUFFER_DIR, "*.npy")))
    for f in npy_files:
        try:
            arr = np.load(f, allow_pickle=True)
            if len(arr) > 0 and isinstance(arr[0], dict):
                payload = arr[0]
                topic = topic_for_type(payload.get("sensor_type"))
                if topic:
                    client.publish(topic, json.dumps(payload))
                    os.remove(f)
        except Exception:
            pass
    # 버퍼 파일 개수 초과 시 오래된 파일 삭제
    npy_files = sorted(glob.glob(os.path.join(BUFFER_DIR, "*.npy")))
    if len(npy_files) > BUFFER_MAX_FILES:
        for f in npy_files[:len(npy_files)-BUFFER_MAX_FILES]:
            try:
                os.remove(f)
            except Exception:
                pass

def topic_for_type(sensor_type):
    return {
        "dht11": TOPIC_DHT,
        "vibration": TOPIC_VIB,
        "sound": TOPIC_SOUND,
        "accel_gyro": TOPIC_IMU,
        "pressure": TOPIC_PRESS,
    }.get(sensor_type)

TOPIC_DHT     = "factory/sensor/dht11"
TOPIC_VIB     = "factory/sensor/vibration"
TOPIC_SOUND   = "factory/sensor/sound"
TOPIC_IMU     = "factory/sensor/accel_gyro"
TOPIC_PRESS   = "factory/sensor/pressure"

# Sampling configuration
# Preferred: specify sampling frequency in Hz (FREQ_*). If you want to keep
# the old interval-in-seconds constants, those are still supported (INTERVAL_*).
# Examples:
#   FREQ_DHT = 1.0   # 1 Hz -> 1.0 second interval
#   FREQ_IMU = 20.0  # 20 Hz -> 0.05 second interval
# Set a value to None to fall back to INTERVAL_* defaults below.
FREQ_DHT     = 1.0
FREQ_VIB     = 16.0
FREQ_SOUND   = 16.0
FREQ_IMU     = 16.0
FREQ_PRESS   = 16.0

# Backward-compatible interval defaults (seconds) — will be overwritten if
# a corresponding FREQ_* value is provided above.
INTERVAL_DHT     = 1.0
INTERVAL_VIB     = 0.1
INTERVAL_SOUND   = 0.1
INTERVAL_IMU     = 0.1
INTERVAL_PRESS   = 0.1

def _hz_to_interval(hz, fallback):
    try:
        if hz is None:
            return fallback
        hz_val = float(hz)
        if hz_val > 0:
            return 1.0 / hz_val
    except Exception:
        pass
    return fallback

# Compute actual intervals (seconds) from frequencies when provided.
INTERVAL_DHT   = _hz_to_interval(FREQ_DHT, INTERVAL_DHT)
INTERVAL_VIB   = _hz_to_interval(FREQ_VIB, INTERVAL_VIB)
INTERVAL_SOUND = _hz_to_interval(FREQ_SOUND, INTERVAL_SOUND)
INTERVAL_IMU   = _hz_to_interval(FREQ_IMU, INTERVAL_IMU)
INTERVAL_PRESS = _hz_to_interval(FREQ_PRESS, INTERVAL_PRESS)

# Choose main-loop sleep and display refresh based on smallest sampling interval.
# We pick a loop sleep that is at most half the smallest interval (so checks are
# responsive), but clamp it to sensible bounds to avoid busy loops or excessive
# CPU usage.
try:
    _MIN_INTERVAL = min(v for v in (INTERVAL_DHT, INTERVAL_VIB, INTERVAL_SOUND, INTERVAL_IMU, INTERVAL_PRESS) if v and v > 0)
except ValueError:
    _MIN_INTERVAL = 0.05
# Sleep between loop iterations (seconds): at least 5ms, at most 50ms.
LOOP_SLEEP = max(0.005, min(0.05, _MIN_INTERVAL / 2.0))
# Display refresh interval (seconds): don't redraw terminal faster than this.
DISPLAY_REFRESH = max(0.02, min(0.1, _MIN_INTERVAL / 2.0))

ADS_ADDR     = 0x48
ADS_GAIN     = 1
ADC_CH_VIB   = 0
ADC_CH_SOUND = 1

stop_flag = False
def handle_stop(sig, frame):
    global stop_flag
    stop_flag = True
signal.signal(signal.SIGINT, handle_stop)
signal.signal(signal.SIGTERM, handle_stop)

# ==============================
# Init devices
# ==============================
def init_ads():
    i2c = busio.I2C(board.SCL, board.SDA)
    ads = ADS1115(i2c, address=ADS_ADDR)
    ads.gain = ADS_GAIN
    ch_vib   = AnalogIn(ads, ADC_CH_VIB)
    ch_sound = AnalogIn(ads, ADC_CH_SOUND)
    return ch_vib, ch_sound

def init_mpu(bus):
    for a in [0x68, 0x69]:
        try:
            bus.read_byte_data(a, 0x75)  # WHO_AM_I
            addr = a
            break
        except Exception:
            continue
    else:
        print("MPU6050 not found")
        return None
    bus.write_byte_data(addr, 0x6B, 0x00)  # wake
    time.sleep(0.05)
    return addr

def read_word_2c(bus, addr, reg):
    hi = bus.read_byte_data(addr, reg)
    lo = bus.read_byte_data(addr, reg + 1)
    val = (hi << 8) | lo
    if val >= 0x8000:
        val = -((65535 - val) + 1)
    return val

def read_mpu(bus, addr):
    ACCEL_SENS, GYRO_SENS = 16384.0, 131.0
    ax = read_word_2c(bus, addr, 0x3B) / ACCEL_SENS
    ay = read_word_2c(bus, addr, 0x3D) / ACCEL_SENS
    az = read_word_2c(bus, addr, 0x3F) / ACCEL_SENS
    gx = read_word_2c(bus, addr, 0x43) / GYRO_SENS
    gy = read_word_2c(bus, addr, 0x45) / GYRO_SENS
    gz = read_word_2c(bus, addr, 0x47) / GYRO_SENS
    return ax, ay, az, gx, gy, gz

def now_ns():
    # Compatible for Python 3.7+
    try:
        return int(time.time_ns())
    except AttributeError:
        return int(time.time() * 1e9)

# ==============================
# Main
# ==============================
def main():
    # 루프 시작 시 버퍼 자동 재전송
    ensure_buffer_dir()
    client = mqtt.Client("sensor_pub_all")
    client.connect(BROKER, PORT, 60)
    client.loop_start()
    resend_buffered(client)
    dht = adafruit_dht.DHT11(board.D4, use_pulseio=False)
    vib_ch, sound_ch = init_ads()
    bus = smbus2.SMBus(1)
    mpu_addr = init_mpu(bus)

    bmp = None
    if HAS_BMP:
        try:
            # Use I2C bus 1
            bmp = BMP085.BMP085(busnum=1)  # default address 0x77
        except Exception as e:
            print("BMP085/BMP180 init error:", repr(e))
            bmp = None

    # ...existing code...

    last_dht = last_vib = last_sound = last_imu = last_press = 0.0

    # Fixed lines (actual last-published values)
    last_line = {
        "dht11":      "DHT11     | (waiting...)",
        "vibration":  "VIBRATION | (waiting...)",
        "sound":      "SOUND     | (waiting...)",
        "accel_gyro": "MPU6050   | (waiting...)",
        "pressure":   "BMP180    | (waiting...)" if bmp else "BMP180    | (not initialized)",
    }

    print("\n=== MOBY Unified Sensor Publisher ===")
    print("Press Ctrl+C to stop.\n")
    # track last display time so we don't redraw terminal every iteration
    last_display = 0.0

    # Use the precomputed loop sleep; keep as local for clarity
    loop_sleep = LOOP_SLEEP

    while not stop_flag:
        now = time.time()

        # ---------- DHT11 ----------
        if now - last_dht >= INTERVAL_DHT:
            try:
                t = dht.temperature
                h = dht.humidity
                if (t is not None) and (h is not None):
                    temperature_c    = round(float(t), 1)
                    humidity_percent = round(float(h), 1)
                    payload = {
                        "sensor_type": "dht11",
                        "sensor_model": "DHT11",
                        "fields": {
                            "temperature_c":    temperature_c,
                            "humidity_percent": humidity_percent
                        },
                        "timestamp_ns": now_ns()
                    }
                    try:
                        client.publish(TOPIC_DHT, json.dumps(payload))
                    except Exception:
                        save_to_buffer("dht11", payload)
                    last_line["dht11"] = "DHT11     | T={:4.1f}C  H={:4.1f}%".format(temperature_c, humidity_percent)
            except Exception as e:
                last_line["dht11"] = "DHT11     | Error: {}".format(e)
            last_dht = now

        # ---------- Vibration ----------
        if now - last_vib >= INTERVAL_VIB:
            try:
                vib_raw  = int(vib_ch.value)
                vib_volt = round(float(vib_ch.voltage), 6)
                payload = {
                    "sensor_type": "vibration",
                    "sensor_model": "SEN0209",
                    "fields": {
                        "vibration_raw":     vib_raw,
                        "vibration_voltage": vib_volt
                    },
                    "timestamp_ns": now_ns()
                }
                try:
                    client.publish(TOPIC_VIB, json.dumps(payload))
                except Exception:
                    save_to_buffer("vibration", payload)
                last_line["vibration"] = "VIBRATION | raw={:5d}  V={:.6f}V".format(vib_raw, vib_volt)
            except Exception as e:
                last_line["vibration"] = "VIBRATION | Error: {}".format(e)
            last_vib = now

        # ---------- Sound ----------
        if now - last_sound >= INTERVAL_SOUND:
            try:
                snd_raw  = int(sound_ch.value)
                snd_volt = round(float(sound_ch.voltage), 6)
                payload = {
                    "sensor_type": "sound",
                    "sensor_model": "AnalogMic_AO",
                    "fields": {
                        "sound_raw":     snd_raw,
                        "sound_voltage": snd_volt
                    },
                    "timestamp_ns": now_ns()
                }
                try:
                    client.publish(TOPIC_SOUND, json.dumps(payload))
                except Exception:
                    save_to_buffer("sound", payload)
                last_line["sound"] = "SOUND     | raw={:5d}  V={:.6f}V".format(snd_raw, snd_volt)
            except Exception as e:
                last_line["sound"] = "SOUND     | Error: {}".format(e)
            last_sound = now

        # ---------- MPU6050 ----------
        if mpu_addr and (now - last_imu >= INTERVAL_IMU):
            try:
                ax, ay, az, gx, gy, gz = read_mpu(bus, mpu_addr)
                ax4, ay4, az4 = round(ax, 4), round(ay, 4), round(az, 4)
                gx4, gy4, gz4 = round(gx, 4), round(gy, 4), round(gz, 4)
                payload = {
                    "sensor_type": "accel_gyro",
                    "sensor_model": "MPU6050",
                    "fields": {
                        "accel_x": ax4,
                        "accel_y": ay4,
                        "accel_z": az4,
                        "gyro_x":  gx4,
                        "gyro_y":  gy4,
                        "gyro_z":  gz4
                    },
                    "timestamp_ns": now_ns()
                }
                try:
                    client.publish(TOPIC_IMU, json.dumps(payload))
                except Exception:
                    save_to_buffer("accel_gyro", payload)
                last_line["accel_gyro"] = (
                    "MPU6050   | Ax={:+.4f} Ay={:+.4f} Az={:+.4f}  "
                    "Gx={:+.4f} Gy={:+.4f} Gz={:+.4f}".format(ax4, ay4, az4, gx4, gy4, gz4)
                )
            except Exception as e:
                last_line["accel_gyro"] = "MPU6050   | Error: {}".format(e)
            last_imu = now

        # ---------- BMP085/BMP180 ----------
        if bmp and (now - last_press >= INTERVAL_PRESS):
            try:
                temp_c     = round(float(bmp.read_temperature()), 2)
                pressure_h = round(float(bmp.read_pressure()) / 100.0, 2)  # Pa -> hPa
                try:
                    altitude_m = round(float(bmp.read_altitude()), 2)
                except Exception:
                    altitude_m = None
                try:
                    slp_h = round(float(bmp.read_sealevel_pressure()) / 100.0, 2)
                except Exception:
                    slp_h = None

                payload = {
                    "sensor_type": "pressure",
                    "sensor_model": "BMP180",
                    "fields": {
                        "temperature_c": temp_c,
                        "pressure_hpa": pressure_h
                    },
                    "timestamp_ns": now_ns()
                }
                if altitude_m is not None:
                    payload["fields"]["altitude_m"] = altitude_m
                if slp_h is not None:
                    payload["fields"]["sea_level_pressure_hpa"] = slp_h

                try:
                    client.publish(TOPIC_PRESS, json.dumps(payload))
                except Exception:
                    save_to_buffer("pressure", payload)

                alt_text = " Alt={:.2f}m".format(altitude_m) if altitude_m is not None else ""
                last_line["pressure"] = "BMP180    | T={:.2f}C  P={:.2f}hPa{}".format(temp_c, pressure_h, alt_text)
            except Exception as e:
                last_line["pressure"] = "BMP180    | Error: {}".format(e)
            last_press = now

        # ---------- Display (rate-limited) ----------
        if now - last_display >= DISPLAY_REFRESH:
            sys.stdout.write("\033[H\033[J")
            print("=== MOBY Edge Sensor Monitor (Live) ===\n")
            print(last_line["dht11"])
            print(last_line["vibration"])
            print(last_line["sound"])
            print(last_line["accel_gyro"])
            print(last_line["pressure"])
            print("\nTime: {}".format(time.strftime("%H:%M:%S")))
            sys.stdout.flush()
            last_display = now

        # sleep a short time to yield CPU but remain responsive to sensor intervals
        time.sleep(loop_sleep)

    # Cleanup
    client.loop_stop()
    client.disconnect()
    bus.close()
    dht.exit()
    print("\nClean exit.")

if __name__ == "__main__":
    main()
