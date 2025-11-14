import RPi.GPIO as GPIO
import time

# --- Pin setup ---
GPIO.setmode(GPIO.BOARD)
AIN1 = 13
AIN2 = 15
PWMA = 12

GPIO.setup(AIN1, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(AIN2, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(PWMA, GPIO.OUT, initial=GPIO.LOW)

# --- PWM setup ---
p = GPIO.PWM(PWMA, 100)  # 100Hz PWM frequency
p.start(0)

try:
    while True:
        # Rotate forward
        GPIO.output(AIN1, GPIO.HIGH)
        GPIO.output(AIN2, GPIO.LOW) 
        p.ChangeDutyCycle(80)  # Motor speed (0~100)
        print("Rotating forward...")
        time.sleep(5)

        # Stop motor
        p.ChangeDutyCycle(0)
        GPIO.output(AIN1, GPIO.LOW)
        GPIO.output(AIN2, GPIO.LOW)
        print("Stopped.")
        time.sleep(2)

except KeyboardInterrupt:
    pass

# --- Cleanup ---
p.stop()
GPIO.cleanup()                                                                                                                                             