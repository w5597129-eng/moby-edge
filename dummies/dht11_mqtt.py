import adafruit_dht
import board
import time
import json
import paho.mqtt.client as mqtt

dhtDevice = adafruit_dht.DHT11(board.D4, use_pulseio=False)

broker = "localhost"
port = 1883
topic = "factory/sensor/dht11"
client_id = "raspberrypi_dht"

client = mqtt.Client(client_id)
client.connect(broker, port, 60)
client.loop_start()

print("MQTT Publisher started.")

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
            json_payload = json.dumps(payload)
            client.publish(topic, json_payload)
            print("Published:", json_payload)
        else:
            print("Checksum failed. Retrying...")

    except RuntimeError as e:
        print("RuntimeError:", e.args[0])
    except Exception as e:
        dhtDevice.exit()
        print("Unexpected error:", e)
        raise e

    time.sleep(2)
