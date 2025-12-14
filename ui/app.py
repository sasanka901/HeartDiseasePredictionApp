import streamlit as st
import pandas as pd
import joblib

# Load model + columns
model = joblib.load("../model/heart_disease_model.joblib")
feature_names = joblib.load("../model/feature_names.joblib")

st.title("Heart Disease Prediction System")

st.write("Enter patient details below:")

# Create input fields dynamically
user_data = {}
for col in feature_names:
    user_data[col] = st.number_input(col, value=0.0)

# Convert to DataFrame
input_df = pd.DataFrame([user_data], columns=feature_names)

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.write("### Prediction:", "Heart Disease" if prediction == 1 else "No Heart Disease")
    st.write("### Probability:", f"{prob*100:.2f}%")
