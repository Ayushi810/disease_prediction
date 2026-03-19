import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("🩺 Disease Prediction System")

# Inputs
preg = st.number_input("Pregnancies")
glucose = st.number_input("Glucose Level")
bp = st.number_input("Blood Pressure")
skin = st.number_input("Skin Thickness")
insulin = st.number_input("Insulin")
bmi = st.number_input("BMI")
dpf = st.number_input("Diabetes Pedigree Function")
age = st.number_input("Age")

# Prediction
if st.button("Predict"):
    data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    result = model.predict(data)

    if result[0] == 1:
        st.error("⚠️ High chance of Diabetes")
    else:
        st.success("✅ Low chance of Diabetes")