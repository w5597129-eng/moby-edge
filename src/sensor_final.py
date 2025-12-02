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

# 버퍼 저장/재전송/정리용 (파일 기반 -> 메모리 기반으로 변경)
import os
import threading
from collections import deque

from inference_interface import (
    InferenceResultMessage,
    WindowMessage,
    WINDOW_SIZE,
    WINDOW_OVERLAP,
    RESULT_TOPIC_ROOT,
    window_topic,
)
# Pressure sensor (BMP085/BMP180)
try:
    import Adafruit_BMP.BMP085 as BMP085  # pip3 install Adafruit-BMP
    HAS_BMP = True
except Exception:
    HAS_BMP = False

# ==============================
# Config
# ==============================
BROKER = "192.168.80.192"
PORT = 1883
# 버퍼 경로
# In-memory publish buffer to avoid SD writes on Raspberry Pi
# stores tuples of (topic, payload_json). Not persistent across restarts.
BUFFER_MAX_ENTRIES = 1000
publish_buffer = deque(maxlen=BUFFER_MAX_ENTRIES)
publish_resender_thread = None

def buffer_publish(client, topic, payload):
    """Try to publish; on failure append to in-memory buffer."""
    try:
        client.publish(topic, json.dumps(payload))
    except Exception:
        try:
            publish_buffer.append((topic, payload))
        except Exception:
            pass

def _resend_worker(client):
    # runs in background, retries buffered publishes
    while not stop_flag:
        try:
            if len(publish_buffer) == 0:
                time.sleep(0.5)
                continue
            topic, payload = publish_buffer.popleft()
            try:
                client.publish(topic, json.dumps(payload))
            except Exception:
                # push back for retry later
                try:
                    publish_buffer.append((topic, payload))
                except Exception:
                    pass
                time.sleep(1.0)
        except Exception:
            time.sleep(1.0)

def start_publish_resender(client):
    global publish_resender_thread
    if publish_resender_thread is None:
        publish_resender_thread = threading.Thread(target=_resend_worker, args=(client,), daemon=True)
        publish_resender_thread.start()

def topic_for_type(sensor_type):
    return {
        "dht11": TOPIC_DHT,
        "vibration": TOPIC_VIB,
        "sound": TOPIC_SOUND,
        "accel_gyro": TOPIC_IMU,
        "pressure": TOPIC_PRESS,
    }.get(sensor_type)


def _on_mqtt_connect(client, userdata, flags, rc):
    if rc == 0:
        try:
            client.subscribe(f"{RESULT_TOPIC_ROOT}/#")
        except Exception:
            pass


def _on_mqtt_message(client, userdata, msg):
    if not msg.topic.startswith(RESULT_TOPIC_ROOT):
        return
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        result = InferenceResultMessage.from_payload(payload)
    except Exception:
        return
    try:
        with inference_results_lock:
            inference_results[result.sensor_type] = {
                "score": result.score,
                "label": result.label,
                "timestamp_ns": result.timestamp_ns or now_ns(),
            }
    except Exception:
        pass

inference_results = {}
inference_results_lock = threading.Lock()

def infer_summary_str(sensor_type):
    with inference_results_lock:
        r = inference_results.get(sensor_type)
    if not r:
        return ""
    s_score = r.get("score")
    s_label = r.get("label")
    try:
        score_s = f"{float(s_score):.4f}"
    except Exception:
        score_s = "n/a"
    label_s = str(s_label) if s_label is not None else "n/a"
    return f"  [INF] score={score_s} label={label_s}"

TOPIC_DHT     = "factory/sensor/dht11"
TOPIC_VIB     = "factory/sensor/vibration"
TOPIC_SOUND   = "factory/sensor/sound"
TOPIC_IMU     = "factory/sensor/accel_gyro"
TOPIC_PRESS   = "factory/sensor/pressure"
TOPIC_IMU_WINDOWS = window_topic("accel_gyro")

# Sampling configuration
# Preferred: specify sampling frequency in Hz (FREQ_*). If you want to keep
# the old interval-in-seconds constants, those are still supported (INTERVAL_*).
# Examples:
#   FREQ_DHT = 1.0   # 1 Hz -> 1.0 second interval
#   FREQ_IMU = 20.0  # 20 Hz -> 0.05 second interval
# Set a value to None to fall back to INTERVAL_* defaults below.
#
# IMPORTANT: FREQ_IMU must match the training data sampling rate (12.8Hz)
# to ensure consistent feature extraction (especially FFT-based features).
FREQ_DHT     = 1.0
FREQ_VIB     = 12.8
FREQ_SOUND   = 12.8
FREQ_IMU     = 12.8  # Must match training data (~78.125ms resample rate)
FREQ_PRESS   = 1.0

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

def _make_mqtt_client(client_id):
    """Create a paho.mqtt.client.Client with fallbacks for different
    paho-mqtt versions. Some installations expect a `callback_api_version`
    parameter or have different constructor signatures; try sensible
    fallbacks and raise the original error if all fail.
    """
    try:
        return mqtt.Client(client_id=client_id)
    except Exception as e:
        # Try explicit callback_api_version for mismatched callback API
        try:
            return mqtt.Client(client_id=client_id, callback_api_version=1)
        except Exception:
            # Try specifying protocol explicitly as a last resort
            try:
                return mqtt.Client(client_id=client_id, userdata=None, protocol=mqtt.MQTTv311)
            except Exception:
                # Re-raise the original exception for visibility
                raise e

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
    # 루프 시작: 메모리 기반 publish 재전송 스레드 시작
    client = _make_mqtt_client("sensor_pub_all")
    client.on_connect = _on_mqtt_connect
    client.on_message = _on_mqtt_message
    client.connect(BROKER, PORT, 60)
    client.loop_start()
    start_publish_resender(client)
    # IMU sampling rate estimate
    sampling_rate_imu = 1.0 / INTERVAL_IMU if INTERVAL_IMU > 0 else 16.0
    buf_len = int(WINDOW_SIZE * sampling_rate_imu) + 2
    overlap_samples = max(int(round(WINDOW_OVERLAP * sampling_rate_imu)), 0)
    accel_x_buf = deque(maxlen=buf_len)
    accel_y_buf = deque(maxlen=buf_len)
    accel_z_buf = deque(maxlen=buf_len)
    gyro_x_buf  = deque(maxlen=buf_len)
    gyro_y_buf  = deque(maxlen=buf_len)
    gyro_z_buf  = deque(maxlen=buf_len)
    axis_buffers = [
        accel_x_buf,
        accel_y_buf,
        accel_z_buf,
        gyro_x_buf,
        gyro_y_buf,
        gyro_z_buf,
    ]
    # Pressure/Temperature buffers for synchronized window inference
    pressure_buf = deque(maxlen=buf_len)
    temperature_buf = deque(maxlen=buf_len)
    
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
                # Only read humidity from DHT11 as requested
                h = dht.humidity
                if h is not None:
                    humidity_percent = round(float(h), 1)
                    payload = {
                        "sensor_type": "dht11",
                        "sensor_model": "DHT11",
                        "fields": {
                            "humidity_percent": humidity_percent
                        },
                        "timestamp_ns": now_ns()
                    }
                    # publish (use memory buffer on failure)
                    buffer_publish(client, TOPIC_DHT, payload)
                    last_line["dht11"] = "DHT11     | H={:4.1f}%".format(humidity_percent)
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
                # publish (use memory buffer on failure)
                buffer_publish(client, TOPIC_VIB, payload)
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
                # publish (use memory buffer on failure)
                buffer_publish(client, TOPIC_SOUND, payload)
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
                # publish (use memory buffer on failure)
                buffer_publish(client, TOPIC_IMU, payload)
                # append to ring buffers for windowed inference
                try:
                    accel_x_buf.append(ax)
                    accel_y_buf.append(ay)
                    accel_z_buf.append(az)
                    gyro_x_buf.append(gx)
                    gyro_y_buf.append(gy)
                    gyro_z_buf.append(gz)
                except Exception:
                    pass
                # enqueue window for inference when buffer full
                try:
                    if all(len(buf) >= buf_len for buf in axis_buffers):
                        window_signals = {
                            'fields_accel_x': list(accel_x_buf),
                            'fields_accel_y': list(accel_y_buf),
                            'fields_accel_z': list(accel_z_buf),
                            'fields_gyro_x':  list(gyro_x_buf),
                            'fields_gyro_y':  list(gyro_y_buf),
                            'fields_gyro_z':  list(gyro_z_buf),
                        }
                        # Include pressure/temperature if available (for consistent feature extraction)
                        if len(pressure_buf) > 0:
                            window_signals['fields_pressure_hpa'] = list(pressure_buf)
                        if len(temperature_buf) > 0:
                            window_signals['fields_temperature_c'] = list(temperature_buf)
                        
                        window_msg = WindowMessage(
                            sensor_type="accel_gyro",
                            sampling_rate_hz=sampling_rate_imu,
                            window_fields=window_signals,
                            timestamp_ns=payload.get("timestamp_ns"),
                            context_payload=payload,
                        )
                        buffer_publish(client, TOPIC_IMU_WINDOWS, window_msg.to_payload())
                        if overlap_samples > 0:
                            for buf in axis_buffers:
                                while len(buf) > overlap_samples:
                                    buf.popleft()
                        else:
                            for buf in axis_buffers:
                                buf.clear()
                except Exception:
                    pass
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

                # publish (use memory buffer on failure)
                buffer_publish(client, TOPIC_PRESS, payload)
                
                # Add to pressure/temperature buffers for synchronized inference
                try:
                    pressure_buf.append(pressure_h)
                    temperature_buf.append(temp_c)
                except Exception:
                    pass

                alt_text = " Alt={:.2f}m".format(altitude_m) if altitude_m is not None else ""
                last_line["pressure"] = "BMP180    | T={:.2f}C  P={:.2f}hPa{}".format(temp_c, pressure_h, alt_text)
            except Exception as e:
                last_line["pressure"] = "BMP180    | Error: {}".format(e)
            last_press = now

        # ---------- Display (rate-limited) ----------
        if now - last_display >= DISPLAY_REFRESH:
            sys.stdout.write("\033[H\033[J")
            print("=== MOBY Edge Sensor Monitor (Live) ===\n")
            # Print sensor lines with latest inference summaries (if any)
            print(last_line["dht11"] + infer_summary_str("dht11"))
            print(last_line["vibration"] + infer_summary_str("vibration"))
            print(last_line["sound"] + infer_summary_str("sound"))
            print(last_line["accel_gyro"] + infer_summary_str("accel_gyro"))
            print(last_line["pressure"] + infer_summary_str("pressure"))
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
