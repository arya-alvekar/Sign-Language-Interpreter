# Sign Language Interpreter Glove

A wearable system that translates sign language gestures into text using flex sensors, Arduino Uno, and machine learning models. The glove is designed to assist communication for the hearing and speech impaired by capturing hand movements and predicting signs in real time.

## Features

Gesture Capture: Flex sensors mounted on the thumb, index, and middle fingers detect finger bending.

Wireless Transmission: Sensor readings are transmitted from the glove to a computer via serial communication.

Machine Learning Integration: Multiple models (Random Forest, Naive Bayes, SVM) are used to classify gestures.

Assistive Communication: Converts recognized signs into text output, enabling easier interaction.

## Hardware Used

Arduino Uno

Flex Sensors (3x – Thumb, Index, Middle fingers)

Rechargeable Battery

Jumper Wires & Breadboard

## Software Used

Arduino IDE (for microcontroller programming)

Python (for data processing & ML model training)

Libraries: scikit-learn, pandas, numpy, matplotlib

## Workflow

Collect flex sensor data through Arduino Uno.

Transmit readings via serial communication.

Preprocess data and train ML models.

Use trained models to classify signs in real time.

Display predicted sign as text output.

## Results

Successfully captured flex sensor readings from multiple fingers.

Achieved accurate classification of signs using ML algorithms.

Demonstrated real-time prediction of hand gestures.

## Future Improvements

Add more flex sensors to cover all five fingers.

Expand dataset for larger vocabulary of signs.

Develop a mobile or web-based interface for real-time translation.

