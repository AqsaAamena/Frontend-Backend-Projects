#Frontend for the model
import streamlit as st
import joblib
import numpy as np

# Load saved model
model = joblib.load("Iris_dataset/best_model.pkl")

st.title("🌸 Iris Flower Classifier")

# User input
sepal_length = st.number_input(
    "Sepal Length", 
    0.0, 10.0, 5.1
    )
sepal_width = st.number_input(
    "Sepal Width",
      0.0, 10.0, 3.5
      )
petal_length = st.number_input(
    "Petal Length",
      0.0, 10.0, 1.4
    )
petal_width = st.number_input(
    "Petal Width",
      0.0, 10.0, 0.2
      )

if st.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]
    st.success(f"Predicted Class: {prediction}")
