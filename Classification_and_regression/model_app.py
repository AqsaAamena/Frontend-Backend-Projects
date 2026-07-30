import streamlit as st
import pandas as pd
import joblib
best_clf = joblib.load("Classification_and_regression/best_classification_model.pkl")
clf_scaler = joblib.load("Classification_and_regression/clf_scaler.pkl")
clf_columns = joblib.load("Classification_and_regression/clf_columns.pkl")
best_reg = joblib.load("Classification_and_regression/best_regression_model.pkl")
reg_scaler = joblib.load("Classification_and_regression/reg_scaler.pkl")
reg_columns = joblib.load("Classification_and_regression/reg_columns.pkl")
st.title("Multi-Model ML App")
problem_type = st.sidebar.selectbox("Select Problem Type", ["Classification", "Regression"])
if problem_type == "Classification":
    st.write("Enter feature values for classification:")
    user_input = {}
    for col in clf_columns:
        user_input[col] = st.number_input(f"{col}", value=0.0)
    input_df = pd.DataFrame([user_input])
    input_scaled = clf_scaler.transform(input_df)
    if st.button("Predict"):
        prediction = best_clf.predict(input_scaled)[0]
        result = "Benign" if prediction == 1 else "Malignant"
        st.success(f"Prediction: {result}")
else:
    st.write("Enter feature values for regression:")
    user_input = {}
    for col in reg_columns:
        user_input[col] = st.number_input(f"{col}", value=0.0)
    input_df = pd.DataFrame([user_input])
    input_scaled = reg_scaler.transform(input_df)
    if st.button("Predict"):
        prediction = best_reg.predict(input_scaled)[0]
        st.success(f"Predicted Charges: {prediction:.2f}")
