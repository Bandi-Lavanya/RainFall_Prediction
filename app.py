from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load the trained model
data = pd.read_csv("rainfall_prediction_data.csv")
data.dropna(inplace=True)
selected_features = ['Temperature', 'Humidity', 'Wind_Speed', 'Pressure', 'Cloud_Cover', 'Previous_Rainfall']
x = data[selected_features]
y = data["Rainfall"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)
model = RandomForestRegressor()
model.fit(x_train, y_train)

@app.route('/')
def index():
    return render_template('mainpage.html')

@app.route('/prediction')
def prediction():
     return render_template('prediction.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        wind_speed = float(request.form['wind_speed'])
        pressure = float(request.form['pressure'])
        cloud_cover = float(request.form['cloud_cover'])
        previous_rainfall = float(request.form['previous_rainfall'])
        
        user_input = np.array([[temperature, humidity, wind_speed, pressure, cloud_cover, previous_rainfall]])
        
        # Make prediction
        predicted_rainfall = model.predict(user_input)
        
        # Redirect to prediction.html with the predicted rainfall value
        return redirect(url_for('result', rainfall=predicted_rainfall[0]))

@app.route('/result/<rainfall>')
def result(rainfall):
    print(rainfall)
    return render_template('pred.html', rainfall=rainfall)

if __name__ == '__main__':
    app.run(debug=True)
