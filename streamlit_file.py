import pickle
import streamlit as st

diagnosa = pickle.load(open('tumor_class.sav', 'rb'))

st.title('Klasifikasi Jenis Kanker Pada Payudara')

radius_mean = st.text_input('Rata Rata Radius Cell Kanker (Mm) : ')
perimeter_mean = st.text_input('Rata Rata Parimeter Cell Kanker (Mm) : ')
area_mean = st.text_input('Rata Rata Luas Area Terdampak Cell Kanker (Mm) : ')
smoothness_mean = st.text_input('Rata Rata Kerataan Pada Cell Kanker (Mm) :')
compactness_mean = st.text_input('Tingkat kepadatan pada cell Kanker (Mm) :')
concavity_mean = st.text_input('Rata Rata Nilai Kecengungan pada Cell Kanker (Mm) :')
fractal_dimension_mean = st.text_input('Rata Rata Nilai Fractal pada Cell Kanker (Mm) :')
texture_mean = st.text_input('Rata Rata Luas Tekstur area terdampak Cell Kanker (Mm) :')


jenis_tumor = ''

if st.button('Diagnosa'):
    tumor_pred = diagnosa.predict([[radius_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, fractal_dimension_mean, texture_mean]])
    
    if(tumor_pred[0] == 0):
        jenis_tumor = 'Tumor Jinak'
    else :
        jenis_tumor ='Tumor Ganas'

    st.success(jenis_tumor)
