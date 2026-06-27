#streamlit run app.py
import streamlit as st
import pickle as pkl
import numpy as np
import pandas as pd

st.title("Car Price Predictor")

df = pd.read_csv('cleaned_data.csv')
arr = sorted(df['company'].unique())
company =  st.selectbox("Enter company",arr)
names = sorted(df[df['company']==company]['name'].unique())
name = st.selectbox("Enter Car name",names)
year = st.number_input("Enter year",min_value=2005,max_value=2026)
kms_driven = st.number_input("Enter kilometers",min_value=500)
fuel_types = sorted(df[df['name']==name]['fuel_type'].unique())
fuel_type = st.selectbox("Enter Fuel type",fuel_types)

if st.button("Predict Price"):
    model = pkl.load(open("model.pkl", "rb+"))
    data = [[company, name, year, kms_driven, fuel_type]]
    columns = ['company', 'name', 'year', 'kms_driven', 'fuel_type']
    df = pd.DataFrame(data, columns=columns)
    st.write(df)
    result = model.predict(df)
    st.write("Predicted price:", round(result[0,0]))