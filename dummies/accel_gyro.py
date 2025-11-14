#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raspberry Pi + MPU-6050 (I2C) publisher
- Shares I2C bus with ADS1115 (no conflict)
- Publishes JSON to MQTT topic "factory/sensor/mpu6050"
- Accel in g, Gyro in deg/s, Temp in °C
"""

import time
import json
import signal
import sys
from math import sqrt
import smbus2
import paho.mqtt.client as mqtt

# ==============================
# User config
# ==============================
DEVICE_ID = "Motor_A_01"
SENSOR_TYPE = "imu"
SENSOR_MODEL = "MPU-6050"

MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "factory/sensor/mpu6050"
MQTT_QOS = 0
PUBLISH_INTERVAL_SEC = 0.2   # 5 Hz
SAMPLES_AVG = 1              # simple averaging window

# ==============================
# MPU-6050 registers & const
# ==============================
MPU_ADDR_CANDIDATES = [0x68, 0x69]    # AD0=GND → 0x68, AD0=VCC → 0x69
REG_PWR_MGMT_1   = 0x6B
REG_SMPLRT_DIV   = 0x19
REG_CONFIG       = 0x1A
REG_GYRO_CONFIG  = 0x1B
REG_ACCEL_CONFIG = 0x1C
REG_INT_STATUS   = 0x3A
REG_ACCEL_XOUT_H = 0x3B
REG_TEMP_OUT_H   = 0x41
REG_GYRO_XOUT_H  = 0x43
REG_WHO_AM_I     = 0x75

ACCEL_SENS_2G = 16384.0     # LSB/g
GYRO_SENS_250 = 131.0       # LSB/(deg/s)

# ==============================
# Helpers
# ==============================
def read_word_2c(bus, addr, reg_h):
    """Read signed 16-bit (two's complement) from high byte register."""
    hi = bus.read_byte_data(addr, reg_h)
    lo = bus.read_byte_data(addr, reg_h + 1)
    val = (hi << 8) | lo
    if val >= 0x8000:
        val = -((65535 - val) + 1)
    return val

def find_mpu(bus):
    """Return detected I2C address from candidates."""
    for a in MPU_ADDR_CANDIDATES:
        try:
            who = bus.read_byte_data(a, REG_WHO_AM_I)
            # On many modules WHO_AM_I returns 0x68
            return a
        except Exception:
            continue
    return None

def mpu_init(bus, addr):
    """Initialize MPU-6050 to a known state."""
    # Wake up device
    bus.write_byte_data(addr, REG_PWR_MGMT_1, 0x00)
    time.sleep(0.05)
    # Set sample rate = Gyro rate / (1 + SMPLRT_DIV); Gyro rate default 8kHz/1kHz depending on DLPF
    bus.write_byte_data(addr, REG_SMPLRT_DIV, 0x07)  # ~1kHz/(1+7)=125Hz base; final depends on DLPF
    # DLPF config: 0x03 → ~44Hz accel/gyro bandwidth (good for noise reduction)
    bus.write_byte_data(addr, REG_CONFIG, 0x03)
    # Gyro full-scale ±250 dps (00)
    bus.write_byte_data(addr, REG_GYRO_CONFIG, 0x00)
    # Accel full-scale ±2g (00)
    bus.write_byte_data(addr, REG_ACCEL_CONFIG, 0x00)
    time.sleep(0.05)

def read_mpu(bus, addr):
    """Read one sample and convert to physical units."""
    ax_raw = read_word_2c(bus, addr, REG_ACCEL_XOUT_H)
    ay_raw = read_word_2c(bus, addr, REG_ACCEL_XOUT_H + 2)
    az_raw = read_word_2c(bus, addr, REG_ACCEL_XOUT_H + 4)
    t_raw  = read_word_2c(bus, addr, REG_TEMP_OUT_H)
    gx_raw = read_word_2c(bus, addr, REG_GYRO_XOUT_H)
    gy_raw = read_word_2c(bus, addr, REG_GYRO_XOUT_H + 2)
    gz_raw = read_word_2c(bus, addr, REG_GYRO_XOUT_H + 4)

    ax = ax_raw / ACCEL_SENS_2G
    ay = ay_raw / ACCEL_SENS_2G
    az = az_raw / ACCEL_SENS_2G
    # Datasheet: Temp in °C = (raw / 340.0) + 36.53
    temp_c = (t_raw / 340.0) + 36.53
    gx = gx_raw / GYRO_SENS_250
    gy = gy_raw / GYRO_SENS_250
    gz = gz_raw / GYRO_SENS_250

    return ax, ay, az, gx, gy, gz, temp_c

# ==============================
# Main
# ==============================
stop_flag = False
def handle_sigint(sig, frame):
    global stop_flag
    stop_flag = True

def main():
    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    # I2C bus 1 on modern Raspberry Pi
    bus = smbus2.SMBus(1)
    addr = find_mpu(bus)
    if addr is None:
        print("ERROR: MPU-6050 not detected at 0x68 or 0x69. Check wiring & i2cdetect -y 1", file=sys.stderr)
        sys.exit(1)

    mpu_init(bus, addr)

    # MQTT
    client = mqtt.Client(client_id=f"{DEVICE_ID}_mpu6050_pub")
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()

    print(f"MPU-6050 publisher started at I2C 0x{addr:02X}. Press Ctrl+C to stop.")
    try:
        while not stop_flag:
            # simple averaging (optional)
            sums = [0.0]*7
            for _ in range(SAMPLES_AVG):
                ax, ay, az, gx, gy, gz, temp_c = read_mpu(bus, addr)
                sums[0] += ax; sums[1] += ay; sums[2] += az
                sums[3] += gx; sums[4] += gy; sums[5] += gz
                sums[6] += temp_c
                if SAMPLES_AVG > 1:
                    time.sleep(PUBLISH_INTERVAL_SEC / max(SAMPLES_AVG,1))

            ax = sums[0]/SAMPLES_AVG
            ay = sums[1]/SAMPLES_AVG
            az = sums[2]/SAMPLES_AVG
            gx = sums[3]/SAMPLES_AVG
            gy = sums[4]/SAMPLES_AVG
            gz = sums[5]/SAMPLES_AVG
            temp_c = sums[6]/SAMPLES_AVG

            payload = {
                "device_id": DEVICE_ID,
                "sensor_type": SENSOR_TYPE,
                "sensor_model": SENSOR_MODEL,
                "i2c_addr": hex(addr),
                "accel_x_g": round(ax, 4),
                "accel_y_g": round(ay, 4),
                "accel_z_g": round(az, 4),
                "gyro_x_dps": round(gx, 2),
                "gyro_y_dps": round(gy, 2),
                "gyro_z_dps": round(gz, 2),
                "temperature_c": round(temp_c, 2),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            client.publish(MQTT_TOPIC, json.dumps(payload), qos=MQTT_QOS, retain=False)
            print(f"Published: {payload}", end="\r")
            time.sleep(PUBLISH_INTERVAL_SEC)
    finally:
        client.loop_stop()
        client.disconnect()
        bus.close()
        print("\nStopped cleanly.")

if __name__ == "__main__":
    main()
