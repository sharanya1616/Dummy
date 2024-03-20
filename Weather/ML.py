import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


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

