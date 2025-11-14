import Adafruit_BMP.BMP085 as BMP085
import time
import sys

# Force stdout encoding to ASCII-safe
sys.stdout.reconfigure(encoding='ascii')

# Initialize BMP085 sensor
sensor = BMP085.BMP085(busnum=1)

print("BMP085 Sensor Reading Start (1s interval). Press Ctrl+C to stop.\n")

try:
    while True:
        # Read sensor values
        temp_c = sensor.read_temperature()
        pressure_pa = sensor.read_pressure()
        altitude_m = sensor.read_altitude()
        sealevel_pa = sensor.read_sealevel_pressure()

        # Print data
        print(
            f"Temp: {temp_c:.2f} C | "
            f"Pressure: {pressure_pa/100:.2f} hPa | "
            f"Altitude: {altitude_m:.2f} m | "
            f"Sea-level Pressure: {sealevel_pa/100:.2f} hPa"
        )

        time.sleep(1)

except KeyboardInterrupt:
    print("\nStopped by user.")
except Exception as e:
    print(f"Error: {e}")
