import serial
import joblib
import numpy as np
import time
from collections import Counter

rf_model = joblib.load('../model/randomforest_model.pkl')
nb_model = joblib.load('../model/naivebayes_model.pkl')
svm_model = joblib.load('../model/svm_model.pkl')
scaler = joblib.load('../model/scaler.pkl')

ser = serial.Serial('COM4', 115200)  
time.sleep(2)  

print("Listening for sensor values...")

try:
    while True:
        line = ser.readline().decode('utf-8').strip()
        print(f"Raw: {line}")

        try:
            values = list(map(float, line.split(',')))
            if len(values) != 3:
                print("Skipping: wrong number of values")
                continue

            scaled = scaler.transform([values])

            predictions = [
                rf_model.predict(scaled)[0],
                nb_model.predict(scaled)[0],
                svm_model.predict(scaled)[0]
            ]

            final_prediction = Counter(predictions).most_common(1)[0][0]
            print(f"Predicted Sign (Ensemble): {final_prediction}\n")

        except ValueError:
            print("Skipping: could not parse numbers")

except KeyboardInterrupt:
    print("Stopped by user")
    ser.close()