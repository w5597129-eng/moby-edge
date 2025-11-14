import adafruit_dht
import board
import time
import json
import paho.mqtt.client as mqtt

dhtDevice = adafruit_dht.DHT11(board.D4, use_pulseio=False)

broker = "localhost"
port = 1883
topic = "factory/sensor/dht11"

client = mqtt.Client("pi_dht11_pub")
client.connect(broker, port, 60)
client.loop_start()

print("DHT11 Publisher started.")

while True:
    try:
        temp = dhtDevice.temperature
        hum = dhtDevice.humidity

        if temp is not None and hum is not None:
            payload = {
                "temperature": temp,
                "humidity": hum,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            client.publish(topic, json.dumps(payload))
            print("Published:", payload)
        else:
            print("Sensor read failed. Retrying...")

    except RuntimeError as e:
        print("RuntimeError:", e.args[0])
    except Exception as e:
        dhtDevice.exit()
        print("Unexpected error:", e)
        raise e

    time.sleep(1)
