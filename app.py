# -*- coding: utf-8 -*-

import streamlit as st 
import pickle
import pandas as pd 
import requests
import urllib.request

model=pickle.load(open("model.pkl","rb"))
st.title('Diabetes Prediction System')

Pregnancies = st.number_input('Pregnancies')
Glucose = st.number_input('Glucose')
BloodPressure = st.number_input('BloodPressure')
SkinThickness = st.number_input('SkinThickness')
Insulin = st.number_input('Insulin')
BMI = st.number_input('BMI')
DiabetesPedigreeFunction = st.number_input('DiabetesPedigreeFunction')
Age = st.slider('Age',0,120)

input_variables = pd.DataFrame([[Pregnancies, Glucose, BloodPressure, SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]],
                                       columns=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'],
                                       dtype=float)

prediction = model.predict(input_variables)[0]
def res():
    if prediction == 0 :
        return "No"
    elif prediction == 1:
        return "Yes"

if st.button('Predict'):
    st.metric(label="Prediction", value=res())
if prediction == 0 :
    st.balloons()
