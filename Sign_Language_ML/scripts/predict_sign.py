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

symbol_map = {
    0: 'Me',
    1: 'okay',
    2: 'One',
    3: 'Victory',
    4: 'Yes',
    5: 'Help',
    6: 'Smile',
    7: 'Hello'
}

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
            print(f"Predicted Sign (Ensemble): {final_prediction}")

            if final_prediction in symbol_map:
                output = symbol_map[final_prediction]
            else:
                binary = format(final_prediction, '03b')[-3:]
                output = f"{binary} - Not a symbol"

            print(f"Interpreted Output: {output}\n")

        except ValueError:
            print("Skipping: could not parse numbers")

except KeyboardInterrupt:
    print("Stopped by user")
    ser.close()