# Import all the necessary libraries
import pandas as pd  # type: ignore
import numpy as np
import joblib  # type: ignore
import streamlit as st  # type: ignore

# Load the model and structure
model = joblib.load("pollution_model.pkl")
model_cols = joblib.load("model_columns.pkl")

# App Title and Introduction
st.markdown("<h1 style='color:#2E86C1;'> Water Pollutants Predictor</h1>", unsafe_allow_html=True)
st.markdown(" Predict key **water pollutants** based on the **Year** and **Station ID** using our trained ML model.")
st.markdown("---")

# Input Section
st.markdown("### Enter Details:")

col1, col2 = st.columns(2)

with col1:
    year_input = st.number_input("Enter Year", min_value=2000, max_value=3000, value=2022)

with col2:
    station_id = st.text_input(" Enter Station ID", value='1')

# Predict Button
if st.button('🔍 Predict Pollutants'):
    if not station_id.strip():
        st.warning(' Please enter a valid Station ID')
    else:
        # Prepare input DataFrame
        input_df = pd.DataFrame({'year': [year_input], 'id': [station_id]})
        input_encoded = pd.get_dummies(input_df, columns=['id'])

        # Align columns with training
        for col in model_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[model_cols]

        # Make Prediction
        predicted_pollutants = model.predict(input_encoded)[0]
        pollutants = ['NH4', 'BSK5', 'Suspended', 'O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']

        # Display Results
        st.markdown("---")
        st.markdown(f"###  Predicted Pollutant Levels for Station **{station_id}** in **{year_input}**:")
        result_col1, result_col2, result_col3 = st.columns(3)

        for i, (p, val) in enumerate(zip(pollutants, predicted_pollutants)):
            col = [result_col1, result_col2, result_col3][i % 3]
            col.metric(label=p, value=f"{val:.2f}")
