import streamlit as st
import pandas as pd
import streamlit as st
import joblib


scaler = joblib.load("scaler.pkl")
le_gender = joblib.load("label_encoder_gender.pkl")
le_diabetic = joblib.load("label_encoder_diabetic.pkl")
le_smoker = joblib.load("label_encoder_smoker.pkl")
model = joblib.load("best_model.pkl")


st.set_page_config(
    page_title="Healthcare Premium Predictor",
    page_icon="✚",
    layout="centered"
)
st.title("Health Insurance Payment Prediction App")
st.write("Enter the detiales below to estimate your insurance payment amount")

with st.form("input form"):

    col1,col2 = st.columns(2)
    
    with col1 :
        age = st.number_input("Age",min_value=0,max_value=100,value=30)
        bmi = st.number_input("BMI",min_value=10.0 , max_value=60.0, value=25.0)
        children = st.number_input("Childrean",max_value=8,min_value=0,value=0)
        
    with col2:
        bloodpressure = st.number_input("Blooad Pressure",min_value=60,max_value=200 ,value = 120)
        gender =st.selectbox("Gender", options=le_gender.classes_)
        diabetic =st.selectbox("Diabetic", options=le_diabetic.classes_)
        smoker = st.selectbox("Smoker", options = le_smoker.classes_)
    
    
    submitted = st.form_submit_button("Predict Paymnet")
    
    
if submitted:

    input_data = pd.DataFrame({
        "age": [age],
        "bmi": [bmi],
        "bloodpressure": [bloodpressure],
        "children": [children],
        "gender_1": [le_gender.transform([gender])[0]],
        "diabetic_1": [le_diabetic.transform([diabetic])[0]],
        "smoker_1": [le_smoker.transform([smoker])[0]]
    })

    # Scale numerical columns
    num_cols = ['age', 'bmi', 'bloodpressure', 'children']

    input_data[num_cols] = scaler.transform(input_data[num_cols])

    # Make prediction
    prediction = model.predict(input_data)[0]

    st.success(
        f"**Estimated Insurance Payment Amount:** ${prediction:,.2f}"
    )
    

