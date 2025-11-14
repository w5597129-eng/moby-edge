#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Sensor Publisher for MOBY Edge Node
- DHT11 (GPIO) -> humidity only
- SEN0209 vibration (ADS1115 A0)
- SZH-EK087 sound (ADS1115 A1)
- MPU-6050 accel/gyro (I2C)
- BMP085/BMP180 pressure+temperature (I2C)
- Fan via TB6612FNG (A-channel), ON at >=30C, OFF at <=29C

ASCII-only prints. Printed values == published values.
"""

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import time, json, signal
import adafruit_dht, board, busio
from adafruit_ads1x15.ads1115 import ADS1115
from adafruit_ads1x15.analog_in import AnalogIn
import smbus2
import paho.mqtt.client as mqtt
import RPi.GPIO as GPIO

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

TOPIC_DHT     = "factory/sensor/dht11"
TOPIC_VIB     = "factory/sensor/vibration"
TOPIC_SOUND   = "factory/sensor/sound"
TOPIC_IMU     = "factory/sensor/accel_gyro"
TOPIC_PRESS   = "factory/sensor/pressure"

INTERVAL_DHT     = 1.0
INTERVAL_VIB     = 1.0
INTERVAL_SOUND   = 1.0
INTERVAL_IMU     = 1.0
INTERVAL_PRESS   = 1.0

ADS_ADDR     = 0x48
ADS_GAIN     = 1
ADC_CH_VIB   = 0
ADC_CH_SOUND = 1

# TB6612FNG pins (BCM)
AIN1 = 27          # TB6612 AIN1
AIN2 = 22          # TB6612 AIN2
PWMA = 18          # TB6612 PWMA (PWM-capable)
STBY = 23          # TB6612 STBY (HIGH to enable)

PWM_FREQ = 100     # Hz
FAN_DUTY_ON = 100  # percent
FAN_ON_TEMP_C  = 30.0
FAN_OFF_TEMP_C = 29.0

stop_flag = False
def handle_stop(sig, frame):
    global stop_flag
    stop_flag = True
signal.signal(signal.SIGINT, handle_stop)
signal.signal(signal.SIGTERM, handle_stop)

# ==============================
# Helpers
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
    try:
        return int(time.time_ns())
    except AttributeError:
        return int(time.time() * 1e9)

# TB6612 fan control (A-channel, one direction)
pwm = None
def fan_init():
    global pwm
    GPIO.setmode(GPIO.BCM)
    GPIO.setup([AIN1, AIN2, PWMA, STBY], GPIO.OUT, initial=GPIO.LOW)
    pwm = GPIO.PWM(PWMA, PWM_FREQ)
    pwm.start(0)
    # standby release
    GPIO.output(STBY, GPIO.HIGH)
    fan_off()

def fan_on(duty=FAN_DUTY_ON):
    # Forward direction: AIN1=HIGH, AIN2=LOW
    GPIO.output(AIN1, GPIO.HIGH)
    GPIO.output(AIN2, GPIO.LOW)
    pwm.ChangeDutyCycle(max(0, min(100, duty)))

def fan_off():
    pwm.ChangeDutyCycle(0)
    GPIO.output(AIN1, GPIO.LOW)
    GPIO.output(AIN2, GPIO.LOW)

def fan_cleanup():
    try:
        fan_off()
        pwm.stop()
    except Exception:
        pass
    try:
        GPIO.output(STBY, GPIO.LOW)
    except Exception:
        pass
    GPIO.cleanup()

# ==============================
# Main
# ==============================
def main():
    # GPIO / TB6612 init
    fan_init()
    fan_state = False  # False=OFF, True=ON

    dht = adafruit_dht.DHT11(board.D4, use_pulseio=False)
    vib_ch, sound_ch = init_ads()
    bus = smbus2.SMBus(1)
    mpu_addr = init_mpu(bus)

    bmp = None
    if HAS_BMP:
        try:
            bmp = BMP085.BMP085(busnum=1)  # default 0x77
        except Exception as e:
            print("BMP085/BMP180 init error:", repr(e))
            bmp = None

    client = mqtt.Client("sensor_pub_all")
    client.connect(BROKER, PORT, 60)
    client.loop_start()

    last_dht = last_vib = last_sound = last_imu = last_press = 0.0

    # Display lines
    last_line = {
        "dht11":      "DHT11     | (waiting...)",
        "vibration":  "VIBRATION | (waiting...)",
        "sound":      "SOUND     | (waiting...)",
        "accel_gyro": "MPU6050   | (waiting...)",
        "pressure":   "BMP180    | (waiting...)" if bmp else "BMP180    | (not initialized)",
        "fan":        "FAN       | OFF",
    }

    print("\n=== MOBY Unified Sensor Publisher ===")
    print("Press Ctrl+C to stop.\n")

    while not stop_flag:
        now = time.time()

        # DHT11 (humidity only)
        if now - last_dht >= INTERVAL_DHT:
            try:
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
                    client.publish(TOPIC_DHT, json.dumps(payload))
                    last_line["dht11"] = "DHT11     | H={:4.1f}%".format(humidity_percent)
            except Exception as e:
                last_line["dht11"] = "DHT11     | Error: {}".format(e)
            last_dht = now

        # Vibration
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
                client.publish(TOPIC_VIB, json.dumps(payload))
                last_line["vibration"] = "VIBRATION | raw={:5d}  V={:.6f}V".format(vib_raw, vib_volt)
            except Exception as e:
                last_line["vibration"] = "VIBRATION | Error: {}".format(e)
            last_vib = now

        # Sound
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
                client.publish(TOPIC_SOUND, json.dumps(payload))
                last_line["sound"] = "SOUND     | raw={:5d}  V={:.6f}V".format(snd_raw, snd_volt)
            except Exception as e:
                last_line["sound"] = "SOUND     | Error: {}".format(e)
            last_sound = now

        # MPU6050
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
                client.publish(TOPIC_IMU, json.dumps(payload))
                last_line["accel_gyro"] = (
                    "MPU6050   | Ax={:+.4f} Ay={:+.4f} Az={:+.4f}  "
                    "Gx={:+.4f} Gy={:+.4f} Gz={:+.4f}".format(ax4, ay4, az4, gx4, gy4, gz4)
                )
            except Exception as e:
                last_line["accel_gyro"] = "MPU6050   | Error: {}".format(e)
            last_imu = now

        # BMP180 (pressure + temperature) + Fan control by temperature
        if bmp and (now - last_press >= INTERVAL_PRESS):
            try:
                temp_c     = round(float(bmp.read_temperature()), 2)
                pressure_h = round(float(bmp.read_pressure()) / 100.0, 2)  # Pa -> hPa
                payload = {
                    "sensor_type": "pressure",
                    "sensor_model": "BMP180",
                    "fields": {
                        "temperature_c": temp_c,
                        "pressure_hpa": pressure_h
                    },
                    "timestamp_ns": now_ns()
                }
                client.publish(TOPIC_PRESS, json.dumps(payload))
                last_line["pressure"] = "BMP180    | T={:.2f}C  P={:.2f}hPa".format(temp_c, pressure_h)

                # Hysteresis control
                if not fan_state and temp_c >= FAN_ON_TEMP_C:
                    fan_on(FAN_DUTY_ON)
                    fan_state = True
                    last_line["fan"] = "FAN       | ON  (duty={}%)".format(FAN_DUTY_ON)
                elif fan_state and temp_c <= FAN_OFF_TEMP_C:
                    fan_off()
                    fan_state = False
                    last_line["fan"] = "FAN       | OFF"
            except Exception as e:
                last_line["pressure"] = "BMP180    | Error: {}".format(e)
            last_press = now

        # Display
        sys.stdout.write("\033[H\033[J")
        print("=== MOBY Edge Sensor Monitor (Live) ===\n")
        print(last_line["dht11"])
        print(last_line["vibration"])
        print(last_line["sound"])
        print(last_line["accel_gyro"])
        print(last_line["pressure"])
        print(last_line["fan"])
        print("\nTime: {}".format(time.strftime("%H:%M:%S")))
        sys.stdout.flush()

        time.sleep(0.05)

    # Cleanup
    try:
        fan_off()
    except Exception:
        pass
    client.loop_stop()
    client.disconnect()
    bus.close()
    dht.exit()
    fan_cleanup()
    print("\nClean exit.")

if __name__ == "__main__":
    main()
