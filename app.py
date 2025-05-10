import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load('loan_approval_rf_model.joblib')
label_encoders = joblib.load('loan_approval_label_encoders.joblib')

# Streamlit app
st.title("💰 Loan Approval Prediction App")

st.write("Fill in the applicant details to predict if the loan will be approved.")

# User inputs using text fields
age = st.text_input("Age", "30")
income = st.text_input("Income ($)", "40000")
credit_score = st.text_input("Credit Score", "700")
loan_amount = st.text_input("Loan Amount ($)", "10000")

# Convert inputs to integers
if st.button("Predict"):
    try:
        age = int(age)
        income = int(income)
        credit_score = int(credit_score)
        loan_amount = int(loan_amount)
        
        # Prepare input data
        input_data = pd.DataFrame({
            'Age': [age],
            'Income ($)': [income],
            'Credit_Score': [credit_score],
            'Loan_Amount ($)': [loan_amount]
        })

        # Make prediction
        prediction = model.predict(input_data)
        predicted_label = label_encoders['Approved'].inverse_transform(prediction)[0]

        st.success(f"Prediction: **{predicted_label}**")

        # Show individual tree predictions (limit to 10 trees)
        st.subheader("🌲 Individual Tree Predictions (Limited to 10 Trees)")
        for i, tree in enumerate(model.estimators_[:10]):
            tree_pred = tree.predict(input_data)
            tree_pred_int = tree_pred.astype(int)
            tree_label = label_encoders['Approved'].inverse_transform(tree_pred_int)[0]
            st.write(f"Tree {i + 1}: {tree_label}")
    
    except ValueError:
        st.error("Please enter valid numeric values for all fields.")
