import RPi.GPIO as GPIO
import time
import requests
import numpy as np
from tensorflow.keras.models import load_model

# Load the saved Keras model
model = load_model("/home/siddhant/Downloads/lr_keras_model.keras")

# GPIO pins setup
GPIO.setmode(GPIO.BCM)
SOIL_MOISTURE_PIN = 23
WATER_PUMP_PIN = 17
GPIO.setup(SOIL_MOISTURE_PIN, GPIO.IN)
GPIO.setup(WATER_PUMP_PIN, GPIO.OUT)

# Function to read soil moisture level
def read_soil_moisture():
    return GPIO.input(SOIL_MOISTURE_PIN)

# Function to predict rain
def predict_rain():
    complete_api_link = "https://api.openweathermap.org/data/2.5/weather?q=dehradun&appid=be49b092b75a0e1237040d7f6e5c8f32"
    api_link = requests.get(complete_api_link)
    api_data = api_link.json()
    temp_city = ((api_data['main']['temp']) - 273.15)
    hmdt = api_data['main']['humidity']
    wind_spd = api_data['wind']['speed']
    pres = (api_data['main']['pressure']) * 100
    temp = np.float64(temp_city)
    humidity = np.float64(hmdt)
    wind_speed = np.float64(wind_spd)
    pressure = np.float64(pres)
    input_data = np.array([[temp, pressure, wind_speed, humidity]])
    predicted_rainfall_prob = model.predict(input_data)
    predicted_rainfall = 1 if predicted_rainfall_prob[0][0] > 0.5 else 0
    return predicted_rainfall

# Function to irrigate if soil is dry and no rain is predicted
def irrigate():
    soil_moisture = read_soil_moisture()
    if soil_moisture == 1:  # Assuming 1 indicates dry soil
        # Predict rain if soil is dry
        if predict_rain() == 0:  # Assuming 0 indicates no rain
            # Turn on water pump
            GPIO.output(WATER_PUMP_PIN, GPIO.HIGH)
            print("Water pump turned on.")
            
            # Check soil moisture continuously until it becomes moist
            while read_soil_moisture() == 1:
                time.sleep(10)  # Adjust interval as needed
                
            # Soil is moist, turn off water pump
            GPIO.output(WATER_PUMP_PIN, GPIO.LOW)
            print("Soil moist. Water pump turned off.")
        else:
            print("Rain predicted. Water pump remains off.")
    else:
        # Soil is already moist, no need for irrigation
        print("Soil moist. No need for irrigation.")

try:
    while True:
        irrigate()        
        # Wait before next reading
        time.sleep(5)  # Adjust interval as needed

except KeyboardInterrupt:
    GPIO.cleanup()
