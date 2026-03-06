from flask import Flask, render_template, request, redirect
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

model = pickle.load(open("diabetes_model.pkl","rb"))

df = pd.read_csv("diabetes_prediction_dataset.csv")

# Login Page
@app.route("/")
def login():
    return render_template("login.html")

# Dashboard
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

# Prediction Page
@app.route("/prediction")
def prediction():
    return render_template("prediction.html")

# Predict
@app.route("/predict", methods=["POST"])
def predict():

    age = float(request.form["age"])
    bmi = float(request.form["bmi"])
    glucose = float(request.form["glucose"])
    bp = float(request.form["bp"])

    data = np.array([[age,bmi,glucose,bp]])

    result = model.predict(data)

    if result[0]==1:
        output="High Risk of Diabetes"
    else:
        output="Low Risk of Diabetes"

    return render_template("prediction.html", prediction_text=output)

# Graph Analysis
@app.route("/analysis")
def analysis():

    plt.figure()
    sns.countplot(x="diabetes", data=df)
    plt.title("Diabetes Distribution")
    plt.savefig("static/graph.png")

    return render_template("dashboard.html")

if __name__ == "__main__":
    app.run(debug=True)