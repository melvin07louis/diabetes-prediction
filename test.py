import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
df = pd.read_csv("diabetes_prediction_dataset.csv")

# Features and target
X = df.drop("diabetes", axis=1)
y = df["diabetes"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

print("Model Accuracy:", acc)

# Save model
pickle.dump(model, open("diabetes_model.pkl", "wb"))