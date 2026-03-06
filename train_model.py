import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("diabetes_prediction_dataset.csv")
df.drop_duplicates(inplace=True)

# Encode categorical columns
gender_enc = LabelEncoder()
smoke_enc = LabelEncoder()
df["gender"] = gender_enc.fit_transform(df["gender"])
df["smoking_history"] = smoke_enc.fit_transform(df["smoking_history"])

# Features and target
X = df.drop("diabetes", axis=1)
y = df["diabetes"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model and encoders
pickle.dump(model, open("diabetes_model.pkl", "wb"))
pickle.dump(gender_enc, open("gender_encoder.pkl", "wb"))
pickle.dump(smoke_enc, open("smoke_encoder.pkl", "wb"))