import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model/diabetes_model.pkl","rb"))

st.title("Diabetes Prediction App")

age = st.slider("Age",1,100)
bmi = st.number_input("BMI")
glucose = st.number_input("Glucose")
bp = st.number_input("Blood Pressure")

if st.button("Predict"):

    data = np.array([[age,bmi,glucose,bp]])

    result = model.predict(data)

    if result[0]==1:
        st.error("High Risk of Diabetes")
    else:
        st.success("Low Risk of Diabetes")
