import pandas as pd
import numpy as np
import streamlit as st
import pickle
from sklearn.linear_model import LinearRegression
loaded_model = pickle.load(open("finalized_model.sav", 'rb'))


st.title("STUDENT TEST SCORE PREDICTION APP")
st.write("Please input the student details in the sidebar")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 1,100)
attendance = st.sidebar.slider("Attendance", 0.1,1.0)

if gender == "Male":
    gender = 0
else:
    gender = 1

user_input = np.array([gender, age, attendance]).reshape(1,-1)
pred = loaded_model.predict(user_input)


st.title(" ")
st.header(f"PREDICTED TEST SCORE IS  --->  {round(pred[0], 2)}")
