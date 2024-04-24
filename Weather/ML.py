import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

# Read the CSV file into a DataFrame
df = pd.read_excel("C:/Desktop/Bestdata.xlsx")

# Drop specified columns
df = df.drop(["MinTemp", "MaxTemp", "Humidity3pm", "Humidity9am", "WindSpeed9am", "WindSpeed3pm", "Pressure9am", "Pressure3pm"], axis=1)

# Remove rows with missing values
df = df.dropna(axis=0)

# Encode 'RainToday' column
le = LabelEncoder()
df['RainToday'] = le.fit_transform(df['RainToday'])

# Define features and target
X = df[['Temperature', 'Pressure', 'WindSpeed', 'Humidity']]
y = df['RainToday']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Logistic Regression
lr = LogisticRegression()
lr.fit(x_train,y_train)
predictions = lr.predict(x_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print(accuracy_score(y_test, predictions))


# API call to get weather data
complete_api_link = "https://api.openweathermap.org/data/2.5/weather?q=cherrapunji&appid=be49b092b75a0e1237040d7f6e5c8f32"
api_link = requests.get(complete_api_link)
api_data = api_link.json()

# Extract relevant weather data and convert to float64
temp_city = ((api_data['main']['temp']) - 273.15)
hmdt = api_data['main']['humidity']
wind_spd = api_data['wind']['speed']
pres = (api_data['main']['pressure']) * 100

# Convert to float64 using numpy
temp = np.float64(temp_city)
humidity = np.float64(hmdt)
wind_speed = np.float64(wind_spd)
pressure = np.float64(pres)

# Reshape data to match model input shape
input_data = np.array([[temp, pressure, wind_speed, humidity]])

# Make predictions using the model
predicted_rainfall_prob = model.predict(input_data)

# Convert predicted probabilities to binary predictions
predicted_rainfall = 1 if predicted_rainfall_prob[0][0] > 0.5 else 0

# Display the predicted rainfall

if predicted_rainfall == 1:
    print("Rain is predicted.")
else:
    print("No rain is predicted.")
