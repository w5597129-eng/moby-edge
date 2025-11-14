# vibration_ads1115_fixed.py
# Raspberry Pi + ADS1115 (I2C, 16-bit) + SEN0209 vibration sensor
# Compatible with new Adafruit CircuitPython ADS1x15 versions

import time
import board
import busio
from adafruit_ads1x15.ads1115 import ADS1115
from adafruit_ads1x15.analog_in import AnalogIn

def main():
    # Initialize I2C
    i2c = busio.I2C(board.SCL, board.SDA)

    # Create ADS1115 instance
    ads = ADS1115(i2c, address=0x48)

    # Set gain (voltage range)
    ads.gain = 1  

    # Read from channel A0
    ch = AnalogIn(ads, 0)

    print("Reading ADS1115 A0 (Ctrl+C to stop)")
    try:
        while True:
            raw = ch.value       # 0 .. 32767
            volt = ch.voltage    # volts
            print(f"ADC={raw:5d}  Voltage={volt:0.4f} V", end="\r")
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == "__main__":
    main()
