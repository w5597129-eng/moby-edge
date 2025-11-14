import time
import board
import adafruit_dht

dhtDevice = adafruit_dht.DHT11(board.D4, use_pulseio=False)

while True:
    try:
        temperature_c = dhtDevice.temperature
        humidity = dhtDevice.humidity
        print(f"Temp: {temperature_c:.1f}C | Humidity: {humidity:.1f}%")
        time.sleep(2)

    except RuntimeError as error:

        print(error.args[0])
        time.sleep(2)
        continue
    except Exception as error:
        dhtDevice.exit()
        raise error
