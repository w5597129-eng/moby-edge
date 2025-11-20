#!/usr/bin/env python3
# -*- coding: us-ascii -*-

import time
import signal
import RPi.GPIO as GPIO

# ======================================
# Config
# ======================================
AIN1 = 27
AIN2 = 22
PWMA = 18
STBY = 23

PWM_FREQ = 100          # Hz

# One conveyor cycle (average, seconds)
CYCLE_SEC = 3.63

# Duty settings
NORMAL_DUTY = 75        # normal speed duty
DIP_DUTY = 40           # slowed-down duty (do not use 0)

# Where in the cycle the dip happens (0.0 ~ 1.0)
# Example: dip starts at 40% of the cycle and lasts 20% of the cycle
DIP_START_RATIO = 0.40
DIP_DURATION_RATIO = 0.20

LOOP_SLEEP_SEC = 0.02   # main loop sleep

stop_flag = False

# ======================================
# SIGINT handler
# ======================================
def handle_sigint(sig, frame):
    global stop_flag
    print("\n[MOTOR] Interrupt received. Stopping...")
    stop_flag = True

signal.signal(signal.SIGINT, handle_sigint)

# ======================================
# GPIO init (BCM mode)
# ======================================
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

# ======================================
# Main
# ======================================
def main():
    init_gpio_bcm()
    pwm = GPIO.PWM(PWMA, PWM_FREQ)
    pwm.start(0)

    print("[MOTOR] Cycle-based step slowdown. Ctrl+C to stop.")
    GPIO.output(STBY, GPIO.HIGH)

    # fixed forward direction
    GPIO.output(AIN1, GPIO.HIGH)
    GPIO.output(AIN2, GPIO.LOW)

    # precompute dip window in seconds
    dip_start_sec = CYCLE_SEC * DIP_START_RATIO
    dip_end_sec = dip_start_sec + CYCLE_SEC * DIP_DURATION_RATIO

    # start at normal duty
    current_duty = NORMAL_DUTY
    pwm.ChangeDutyCycle(current_duty)
    print(f"[MOTOR] duty={current_duty}% (normal start)")

    cycle_t0 = time.time()

    try:
        while not stop_flag:
            now = time.time()
            elapsed = now - cycle_t0

            # wrap elapsed into [0, CYCLE_SEC)
            phase = elapsed % CYCLE_SEC

            # decide duty based on phase position in the cycle
            if dip_start_sec <= phase < dip_end_sec:
                target_duty = DIP_DUTY
            else:
                target_duty = NORMAL_DUTY

            if target_duty != current_duty:
                current_duty = target_duty
                pwm.ChangeDutyCycle(current_duty)
                if current_duty == DIP_DUTY:
                    print(f"[MOTOR] duty={current_duty}% (dip)")
                else:
                    print(f"[MOTOR] duty={current_duty}% (normal)")

            time.sleep(LOOP_SLEEP_SEC)

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
