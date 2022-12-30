import streamlit as st
import pickle
import numpy as np
from sklearn import * 
import pandas as pd


# import the model
pipe_rf = pickle.load(open('rf1.pkl','rb'))
# import the dataset
df = pickle.load(open('data.pkl','rb'))


st.title("Laptop Predictor")

# brand
company = st.selectbox('Brand',df['Company'].unique())

# type of laptop
type = st.selectbox('Type',df['TypeName'].unique())

# Ram
ram = st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

# weight
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen',['No','Yes'])

# IPS
ips = st.selectbox('IPS',['No','Yes'])

#cpu
cpu = st.selectbox('CPU',df['Cpu brand'].unique())

hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

gpu = st.selectbox('GPU',df['Gpu brand'].unique())

os = st.selectbox('OS',df['os'].unique())


if st.button('Predict Price'):
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    query = np.array([company,type,ram,weight,touchscreen,ips,cpu,hdd,ssd,gpu,os])

    query = query.reshape(1,11)
    st.write(pipe_rf.predict(query)[0])
    st.title("The predicted price " + str(int(pipe_rf.predict(query)[0])))

