import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
data = pd.read_csv('C:\\Users\\Hp\\OneDrive\\Desktop\\C++\\sign_language_glove_2\\data\\sensor_data.csv', header=None)
data.columns = ['thumb', 'index', 'middle', 'label']

# Print class distribution to understand the issue
print("Label distribution:")
print(data['label'].value_counts())

# Ensure all values are numeric
for col in ['thumb', 'index', 'middle', 'label']:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Drop rows with NaN values
data = data.dropna()

# Features and Labels
X = data[['thumb', 'index', 'middle']].values
y = data['label'].values

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define classifiers
classifiers = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'NaiveBayes': GaussianNB(),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42)
}

# Train, evaluate, and save each model
for name, clf in classifiers.items():
    print(f"\n--- Training {name} ---")
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    print(f"\n{name} Accuracy:", accuracy_score(y_test, y_pred))
    print(f"\n{name} Classification Report:\n", classification_report(y_test, y_pred))
    joblib.dump(clf, f'{name.lower()}_model.pkl')

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')
print("\nAll models and scaler have been saved.")