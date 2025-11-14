#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import signal
import RPi.GPIO as GPIO

# ==============================
# Config
# ==============================
AIN1 = 27
AIN2 = 22
PWMA = 18
STBY = 23

PWM_FREQ = 100
DUTY = 50
RUN_SEC = 5
STOP_SEC = 5

stop_flag = False

# ==============================
# Graceful shutdown
# ==============================
def handle_sigint(sig, frame):
    global stop_flag
    print("\n[MOTOR] Interrupt received. Stopping motor...")
    stop_flag = True

signal.signal(signal.SIGINT, handle_sigint)

# ==============================
# GPIO init (robust)
# ==============================
def init_gpio_bcm():
    GPIO.setwarnings(False)
    mode = GPIO.getmode()
    if mode is not None and mode != GPIO.BCM:
        GPIO.cleanup()
    try:
        GPIO.setmode(GPIO.BCM)
    except ValueError:
        GPIO.cleanup()
        GPIO.setmode(GPIO.BCM)

    GPIO.setup(AIN1, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(AIN2, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(PWMA, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(STBY, GPIO.OUT, initial=GPIO.LOW)

# ==============================
# Main
# ==============================
def main():
    init_gpio_bcm()
    pwm = GPIO.PWM(PWMA, PWM_FREQ)
    pwm.start(0)

    print("[MOTOR] Running. Ctrl+C to stop.")
    GPIO.output(STBY, GPIO.HIGH)

    try:
        while not stop_flag:
            # Forward
            GPIO.output(AIN1, GPIO.HIGH)
            GPIO.output(AIN2, GPIO.LOW)
            pwm.ChangeDutyCycle(DUTY)
            print("[MOTOR] Forward...")
            time.sleep(RUN_SEC)

            # Stop
            pwm.ChangeDutyCycle(0)
            GPIO.output(AIN1, GPIO.LOW)
            GPIO.output(AIN2, GPIO.LOW)
            print("[MOTOR] Stop.")
            time.sleep(STOP_SEC)

    finally:
        pwm.ChangeDutyCycle(0)
        pwm.stop()
        GPIO.output(AIN1, GPIO.LOW)
        GPIO.output(AIN2, GPIO.LOW)
        GPIO.output(STBY, GPIO.LOW)
        GPIO.cleanup()
        print("[MOTOR] Clean exit.")

if __name__ == "__main__":
    main()
