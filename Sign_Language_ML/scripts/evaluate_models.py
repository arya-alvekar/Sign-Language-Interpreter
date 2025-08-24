import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('data/sensor_data.csv', header=None)
data.columns = ['thumb', 'index', 'middle', 'label']

for col in ['thumb', 'index', 'middle', 'label']:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data = data.dropna()

X = data[['thumb', 'index', 'middle']].values
y = data['label'].values

scaler = joblib.load('model/scaler.pkl')
X_scaled = scaler.transform(X)

model_files = {
    'RandomForest': 'model/randomforest_model.pkl',
    'NaiveBayes': 'model/naivebayes_model.pkl',
    'SVM': 'model/svm_model.pkl'
}

for name, path in model_files.items():
    print(f"\n--- Evaluating {name} ---")
    model = joblib.load(path)
    y_pred = model.predict(X_scaled)

    print(f"{name} Accuracy: {accuracy_score(y, y_pred):.4f}")
    print(f"\n{name} Classification Report:\n{classification_report(y, y_pred)}")

    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()