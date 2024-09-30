import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor#for linear regression analysis.
from sklearn.metrics import mean_squared_error#to evaluate the performance of the regression model.
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
data=pd.read_csv("rainfall_prediction_data.csv")
data.isnull().sum()
data.dropna(inplace=True)
data.dropna(inplace=False)
selected_features=['Temperature','Humidity','Wind_Speed','Pressure', 'Cloud_Cover','Previous_Rainfall' ]
x=data[selected_features]
y=data["Rainfall"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=5)
model=RandomForestRegressor()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error:{mae}")
import numpy as np
# Location = (input("Enter Location: "))
temperature = float(input("Enter Temperature: "))
humidity = float(input("Enter Humidity: "))
wind_speed = float(input("Enter Wind Speed: "))
pressure = float(input("Enter Pressure: "))
cloud_cover = float(input("Enter Cloud Cover: "))
previous_rainfall = float(input("Enter Previous Rainfall: "))
user_input = np.array([[temperature, humidity, wind_speed, pressure, cloud_cover, previous_rainfall]])

# Make prediction
predicted_rainfall = model.predict(user_input)

# Output the predicted rainfall
print("Predicted Rainfall:", predicted_rainfall[0])

def prediction():
    return render_template