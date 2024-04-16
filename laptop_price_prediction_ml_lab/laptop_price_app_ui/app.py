import streamlit as st
import pickle
import numpy as np

from sklearn.preprocessing import OneHotEncoder


# import the model
pipe = pickle.load(open('pipe_random.pkl','rb'))
data = pickle.load(open('laptop_price_data.pkl','rb'))


st.title("Laptop Predictor")

# brand
company = st.selectbox('Brand',data['Company'].unique())

# processor_brand
processor_brand = st.selectbox('Processor_brand',data['processor_brand'].unique())

# Ram
ram = st.number_input('RAM(in GB)')

#ram_type
ram_type = st.selectbox('ram_type',data['ram_type'].unique())

# processor_name
processor_name = st.selectbox('processor_name',data['processor_name'].unique())

# processor_genration
processor_generation = st.selectbox('processor_generation',data['processor_gnrtn'].unique())

#cpu
ssd = st.selectbox('ssd',data['ssd'].unique())

# os
os = st.selectbox('OS',data['os'].unique())

#graphic card
graphic_card_gb = st.selectbox('Graphic Card (in GB)', data['graphic_card_gb'].unique())

# weight
weight = st.selectbox('weight',data['weight'].unique())

#warranty
warranty = st.selectbox('warranty',data['warranty'].unique())

# Touchscreen
touchscreen = st.selectbox('touchscreen',data['Touchscreen'].unique())

# msoffice
msoffice = st.selectbox('msoffice',data['msoffice'].unique())


if st.button('Predict Price'):
    encoder = OneHotEncoder(handle_unknown='ignore')

    query = np.array([company,processor_brand,processor_name,processor_generation,ram,ram_type,ssd,os,graphic_card_gb,weight,warranty,touchscreen,msoffice])

    query = query.reshape(1,-1)
    st.title("The predicted price of this configuration is " + str(int(np.exp(pipe.predict(query)[0]))))
