#!/usr/bin/env python3
# -*- coding: us-ascii -*-
"""
IR conveyor cycle counter (pigpio interrupt first, RPi.GPIO polling fallback)
- ASCII-only output; safe stdout for Thonny.
"""

import sys
try:
    sys.stdout.reconfigure(encoding="ascii", errors="backslashreplace")
except Exception:
    pass

import time
import statistics

IR_PIN = 17                # D0 -> BCM 17 (physical 11)
DEAD_TIME_MS = 200         # dead time after a valid hit
AVG_WINDOW = 10            # moving average window
PRINT_EVERY = 1            # print every N cycles
MARKS_PER_REV = 1          # marks per revolution for RPM
ENABLE_RPM = True
ENABLE_BELT_SPEED = False
PULLEY_CIRCUMFERENCE_M = 0.50

last_hit_ns = None
dead_until_ns = 0
cycle_times_ms = []
cycle_count = 0

def now_ns():
    return time.monotonic_ns()

def record_hit(t_ns):
    global last_hit_ns, dead_until_ns, cycle_count, cycle_times_ms
    if t_ns < dead_until_ns:
        return
    if last_hit_ns is None:
        last_hit_ns = t_ns
        dead_until_ns = t_ns + DEAD_TIME_MS * 1_000_000
        return
    dt_ms = (t_ns - last_hit_ns) / 1_000_000.0
    last_hit_ns = t_ns
    dead_until_ns = t_ns + DEAD_TIME_MS * 1_000_000
    # reject unrealistically small intervals
    if dt_ms < DEAD_TIME_MS * 1.2:
        return
    cycle_count += 1
    cycle_times_ms.append(dt_ms)
    if len(cycle_times_ms) > AVG_WINDOW:
        cycle_times_ms = cycle_times_ms[-AVG_WINDOW:]
    if cycle_count % PRINT_EVERY == 0:
        avg_ms = statistics.mean(cycle_times_ms) if cycle_times_ms else float("nan")
        msg = {
            "cycles": cycle_count,
            "last_cycle_ms": round(dt_ms, 2),
            "avg_cycle_ms": round(avg_ms, 2) if avg_ms == avg_ms else None
        }
        if ENABLE_RPM and avg_ms == avg_ms:
            rps = (1000.0 / avg_ms) / float(MARKS_PER_REV)
            rpm = rps * 60.0
            msg["rpm"] = round(rpm, 2)
            if ENABLE_BELT_SPEED:
                msg["belt_m_per_s"] = round(rps * PULLEY_CIRCUMFERENCE_M, 3)
        print(msg)

def run_pigpio():
    import pigpio
    pi = pigpio.pi()
    if not pi.connected:
        raise RuntimeError("pigpio daemon is not running")
    # Measure idle level quickly
    pi.set_mode(IR_PIN, pigpio.INPUT)
    # Try default assumption: idle LOW -> rising pulses on detect
    # If your module idles HIGH, we swap later based on read.
    idle = 0
    sample = []
    t0 = time.time()
    while time.time() - t0 < 0.2:
        sample.append(pi.read(IR_PIN))
        time.sleep(0.005)
    if sample:
        idle = 1 if sum(sample) >= (len(sample)/2.0) else 0

    if idle == 0:
        pi.set_pull_up_down(IR_PIN, pigpio.PUD_DOWN)
        edge = pigpio.RISING_EDGE
        label = "PUD_DOWN + RISING"
    else:
        pi.set_pull_up_down(IR_PIN, pigpio.PUD_UP)
        edge = pigpio.FALLING_EDGE
        label = "PUD_UP + FALLING"

    # Glitch filter: ignore pulses shorter than x microseconds (debounce)
    # For 80 ms debounce in RPi.GPIO, a small glitch filter is enough here (e.g., 200 us).
    pi.set_glitch_filter(IR_PIN, 200)

    print("pigpio mode. Idle={} | {}".format(idle, label))

    def cbf(gpio, level, tick):
        # tick is in microseconds (wraps at ~1h). Use monotonic_ns for robust timing.
        record_hit(now_ns())

    cb = pi.callback(IR_PIN, edge, cbf)

    print("Interrupt mode active (pigpio). Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            cb.cancel()
        except Exception:
            pass
        pi.set_glitch_filter(IR_PIN, 0)
        pi.stop()
        print("Stopped (pigpio).")

def run_polling():
    import RPi.GPIO as GPIO
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    # Measure idle
    GPIO.setup(IR_PIN, GPIO.IN)
    vals = []
    t0 = time.time()
    while time.time() - t0 < 0.3:
        vals.append(GPIO.input(IR_PIN))
        time.sleep(0.01)
    idle = 1 if (vals and sum(vals) >= len(vals)/2.0) else 0

    if idle == 0:
        pud = GPIO.PUD_DOWN
        edge_str = "RISING"
    else:
        pud = GPIO.PUD_UP
        edge_str = "FALLING"

    GPIO.setup(IR_PIN, GPIO.IN, pull_up_down=pud)
    prev = GPIO.input(IR_PIN)
    print("Fallback: polling mode. Idle={} | {}".format(idle, "PUD_DOWN" if pud==GPIO.PUD_DOWN else "PUD_UP"))

    try:
        while True:
            cur = GPIO.input(IR_PIN)
            if edge_str == "RISING":
                if prev == 0 and cur == 1:
                    record_hit(now_ns())
            else:
                if prev == 1 and cur == 0:
                    record_hit(now_ns())
            prev = cur
            time.sleep(0.001)  # 1 ms polling
    except KeyboardInterrupt:
        pass
    finally:
        GPIO.cleanup()
        print("Stopped (polling).")

if __name__ == "__main__":
    # Try pigpio first; if unavailable, fall back to polling
    try:
        import pigpio  # import test only
        run_pigpio()
    except Exception as e:
        print("pigpio not available or failed: {}. Using polling...".format(str(e)))
        run_polling()
