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
import pickle, threading, queue
from collections import deque
# feature extraction helper from repo
from feature_extractor import extract_features, SENSOR_FIELDS, WINDOW_SIZE, USE_FREQUENCY_DOMAIN

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

# ==============================
# Model / Inference (async worker)
# ==============================
MODEL_PATH = "models/isolation_forest.pkl"
SCALER_PATH = "models/scaler_if.pkl"

def load_model_and_scaler():
    model = None
    scaler = None
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        print("Model load error:", e)
    try:
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
    except Exception as e:
        print("Scaler load error:", e)
    return model, scaler

inference_q = queue.Queue(maxsize=512)
inference_stop = threading.Event()

def inference_worker(client, model, scaler, stop_event):
    while not stop_event.is_set():
        try:
            sensor_type, payload, window_signals, sampling_rate = inference_q.get(timeout=1.0)
        except queue.Empty:
            continue
        try:
            feats = []
            for field in SENSOR_FIELDS:
                sig = window_signals.get(field)
                if sig is None or len(sig) < 2:
                    feat_len = 11 if USE_FREQUENCY_DOMAIN else 5
                    feats.extend([0.0] * feat_len)
                else:
                    f = extract_features(np.array(sig), sampling_rate, use_freq_domain=USE_FREQUENCY_DOMAIN)
                    feats.extend(f)
            X = np.array(feats).reshape(1, -1)
            if scaler is not None:
                try:
                    Xs = scaler.transform(X)
                except Exception:
                    Xs = X
            else:
                Xs = X
            result_score = None
            result_label = None
            if model is not None:
                try:
                    result_score = float(model.score_samples(Xs)[0])
                except Exception:
                    pass
                try:
                    result_label = int(model.predict(Xs)[0])
                except Exception:
                    pass
            if result_score is not None:
                payload.setdefault("fields", {})["anomaly_score"] = result_score
            if result_label is not None:
                payload.setdefault("fields", {})["anomaly_label"] = result_label
            base_topic = topic_for_type(sensor_type)
            out_topic = f"{base_topic}/inference" if base_topic else None
            if out_topic:
                try:
                    client.publish(out_topic, json.dumps(payload))
                except Exception:
                    # if publish fails, save to buffer
                    save_to_buffer(sensor_type, payload)
        except Exception as e:
            print("Inference worker error:", e)
        finally:
            try:
                inference_q.task_done()
            except Exception:
                pass

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
    # -----------------------------
    # Optional YAML-based configuration (flexible loader)
    # Looks in multiple places: `src/sensor_config.yaml`, `../config/sensor_config.yaml`,
    # and `./config/sensor_config.yaml` (repo root). Supports both flat and nested keys
    # (legacy formats and the attachment's structure).
    # If PyYAML is not installed the loader will warn and skip config.
    # -----------------------------
    def _load_yaml_config():
        candidates = [
            os.path.join(os.path.dirname(__file__), 'sensor_config.yaml'),
            os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config', 'sensor_config.yaml')),
            os.path.abspath(os.path.join(os.getcwd(), 'config', 'sensor_config.yaml')),
        ]
        cfg_path = None
        for p in candidates:
            if os.path.exists(p):
                cfg_path = p
                break
        if not cfg_path:
            return {}
        try:
            import yaml
        except Exception:
            print('Warning: PyYAML not installed; skipping', cfg_path)
            return {}
        try:
            with open(cfg_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print('Warning: failed to load', cfg_path, ':', e)
            return {}

    def _apply_config(cfg):
        global BROKER, PORT, BUFFER_DIR, BUFFER_MAX_FILES
        global TOPIC_DHT, TOPIC_VIB, TOPIC_SOUND, TOPIC_IMU, TOPIC_PRESS
        global MODEL_PATH, SCALER_PATH
        global FREQ_DHT, FREQ_VIB, FREQ_SOUND, FREQ_IMU, FREQ_PRESS
        global INTERVAL_DHT, INTERVAL_VIB, INTERVAL_SOUND, INTERVAL_IMU, INTERVAL_PRESS
        global LOOP_SLEEP, DISPLAY_REFRESH, _MIN_INTERVAL
        global ADS_ADDR, ADS_GAIN, ADC_CH_VIB, ADC_CH_SOUND

        # mqtt top-level or nested
        broker = cfg.get('broker')
        if broker is None:
            broker = (cfg.get('mqtt') or {}).get('broker')
        if broker:
            BROKER = broker
        port = cfg.get('port')
        if port is None:
            port = (cfg.get('mqtt') or {}).get('port')
        if port:
            try:
                PORT = int(port)
            except Exception:
                pass

        # buffer
        bdir = cfg.get('buffer_dir') or (cfg.get('buffer') or {}).get('directory')
        if bdir:
            BUFFER_DIR = bdir
        bmax = cfg.get('buffer_max_files') or (cfg.get('buffer') or {}).get('max_files')
        if bmax:
            try:
                BUFFER_MAX_FILES = int(bmax)
            except Exception:
                pass

        # topics - support both flat and mqtt.topics
        topics = cfg.get('topics') or (cfg.get('mqtt') or {}).get('topics') or {}
        # mapping tolerant to different key names
        TOPIC_DHT = topics.get('dht') or topics.get('dht11') or TOPIC_DHT
        TOPIC_VIB = topics.get('vibration') or TOPIC_VIB
        TOPIC_SOUND = topics.get('sound') or TOPIC_SOUND
        TOPIC_IMU = topics.get('imu') or topics.get('accel_gyro') or TOPIC_IMU
        TOPIC_PRESS = topics.get('pressure') or TOPIC_PRESS

        # model/scaler
        MODEL_PATH = cfg.get('model_path') or cfg.get('model') or MODEL_PATH
        SCALER_PATH = cfg.get('scaler_path') or cfg.get('scaler') or SCALER_PATH

        # frequencies: support `freq` map or nested `sampling` structure
        freqs = cfg.get('freq') or {}
        sampling = cfg.get('sampling') or cfg.get('sampling_rate') or {}
        # helper to read frequency
        def _read_freq(name, alt_name=None):
            v = freqs.get(name)
            if v is None and alt_name:
                v = freqs.get(alt_name)
            if v is None:
                # check nested sampling
                s = sampling.get(name) or sampling.get(alt_name or name) or {}
                if isinstance(s, dict):
                    v = s.get('frequency_hz') or s.get('frequency')
            return v

        f = _read_freq('dht', 'dht11')
        if f is None:
            f = _read_freq('dht11')
        FREQ_DHT = f if f is not None else FREQ_DHT
        FREQ_VIB = _read_freq('vibration') or FREQ_VIB
        FREQ_SOUND = _read_freq('sound') or FREQ_SOUND
        FREQ_IMU = _read_freq('imu', 'accel_gyro') or FREQ_IMU
        FREQ_PRESS = _read_freq('pressure') or FREQ_PRESS

        # recompute intervals
        INTERVAL_DHT   = _hz_to_interval(FREQ_DHT, INTERVAL_DHT)
        INTERVAL_VIB   = _hz_to_interval(FREQ_VIB, INTERVAL_VIB)
        INTERVAL_SOUND = _hz_to_interval(FREQ_SOUND, INTERVAL_SOUND)
        INTERVAL_IMU   = _hz_to_interval(FREQ_IMU, INTERVAL_IMU)
        INTERVAL_PRESS = _hz_to_interval(FREQ_PRESS, INTERVAL_PRESS)

        try:
            _MIN_INTERVAL = min(v for v in (INTERVAL_DHT, INTERVAL_VIB, INTERVAL_SOUND, INTERVAL_IMU, INTERVAL_PRESS) if v and v > 0)
        except Exception:
            pass
        # allow display tuning if provided
        disp = cfg.get('display') or {}
        min_ms = disp.get('loop_sleep_min_ms') or 5
        max_ms = disp.get('loop_sleep_max_ms') or 50
        min_disp = disp.get('display_refresh_min_ms') or 20
        max_disp = disp.get('display_refresh_max_ms') or 100
        LOOP_SLEEP = max(float(min_ms)/1000.0, min(float(max_ms)/1000.0, _MIN_INTERVAL / 2.0))
        DISPLAY_REFRESH = max(float(min_disp)/1000.0, min(float(max_disp)/1000.0, _MIN_INTERVAL / 2.0))

        # ADS1115 / ADC
        ads = cfg.get('ads') or cfg.get('ads1115') or {}
        if 'addr' in ads:
            try:
                ADS_ADDR = int(ads.get('addr'))
            except Exception:
                # handle hex string like 0x48
                try:
                    ADS_ADDR = int(str(ads.get('addr')), 0)
                except Exception:
                    pass
        ADS_GAIN = ads.get('gain', ADS_GAIN)
        try:
            ADC_CH_VIB = int(ads.get('ch_vib') or ads.get('channel_vibration') or ADC_CH_VIB)
            ADC_CH_SOUND = int(ads.get('ch_sound') or ads.get('channel_sound') or ADC_CH_SOUND)
        except Exception:
            pass


    _CFG = _load_yaml_config()
    if _CFG:
        _apply_config(_CFG)

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
                    if len(accel_x_buf) >= buf_len:
                        window_signals = {
                            'fields_accel_x': list(accel_x_buf),
                            'fields_accel_y': list(accel_y_buf),
                            'fields_accel_z': list(accel_z_buf),
                            'fields_gyro_x':  list(gyro_x_buf),
                            'fields_gyro_y':  list(gyro_y_buf),
                            'fields_gyro_z':  list(gyro_z_buf),
                        }
                        payload_for_infer = payload.copy()
                        try:
                            inference_q.put_nowait(("accel_gyro", payload_for_infer, window_signals, sampling_rate_imu))
                        except queue.Full:
                            # drop if queue is full
                            pass
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
    # stop inference worker
    try:
        inference_stop.set()
        inference_thread.join(timeout=1.0)
    except Exception:
        pass
    client.loop_stop()
    client.disconnect()
    bus.close()
    dht.exit()
    print("\nClean exit.")

if __name__ == "__main__":
    main()
