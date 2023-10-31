import pickle
import streamlit as st

diagnosa = pickle.load(open('diabetes.sav', 'rb'))

st.title('Diagnosa Resiko Anda Terkena Diabetes')

Pregnancies = st.number_input('Masa Kehamilan anda (bulan)')
Glucose = st.number_input('Tingkat Gulkosa Dalam Tubuh')
BloodPressure = st.number_input('Tekanan Darah')
SkinThickness = st.number_input('Ketebalan Kulit Terhadap Lemak')
Insulin = st.number_input('Tingkat Insulin')
BMI = st.number_input('Tingkat BMI')
DiabetesPedigreeFunction = st.number_input('Tingkat DPF')
Age = st.number_input('Usia Anda')

resiko_diabetes = ''

if st.button('Diagnosa'):
    resiko = diagnosa.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    
    if(resiko[0] == 0):
        resiko_diabetes = 'Anda Tidak Beresiko Diabetes'
    else :
        resiko_diabetes ='Anda Beresiko Terkena Diabetes'

    st.success(resiko_diabetes)