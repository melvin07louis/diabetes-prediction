from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
import seaborn as sns

app = Flask(__name__)
app.secret_key = "supersecretkey"  # required for session

# Load model and encoders
model = pickle.load(open("diabetes_model.csv", "rb"))
gender_enc = pickle.load(open("gender_encoder.pkl", "rb"))
smoke_enc = pickle.load(open("smoke_encoder.pkl", "rb"))

# Dummy user (for login)
USER_EMAIL = "admin@gmail.com"
USER_PASSWORD = "1234"

# ---------------- LOGIN PAGE ----------------
@app.route("/", methods=["GET", "POST"])
def login():
    error = ""
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        if email == USER_EMAIL and password == USER_PASSWORD:
            session["user"] = email
            return redirect(url_for("home"))
        else:
            error = "Invalid email or password"
    return render_template("login.html", error=error)

# ---------------- HOME PAGE ----------------
@app.route("/home", methods=["GET", "POST"])
def home():
    if "user" not in session:
        return redirect(url_for("login"))

    result = ""
    if request.method == "POST":
        try:
            gender = request.form["gender"]
            age = int(request.form["age"])
            hypertension = int(request.form["hypertension"])
            heart_disease = int(request.form["heart_disease"])
            smoking = request.form["smoking"]
            bmi = float(request.form["bmi"])
            hba1c = float(request.form["hba1c"])
            glucose = float(request.form["glucose"])

            g = gender_enc.transform([gender])[0]
            s = smoke_enc.transform([smoking])[0]

            input_data = np.array([[g, age, hypertension, heart_disease, s, bmi, hba1c, glucose]])
            pred = model.predict(input_data)[0]

            if pred == 1:
                result = "High Risk of Diabetes"
            else:
                result = "Low Risk of Diabetes"
        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template("home.html", result=result)

# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)