import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler 
import pandas as pd
import streamlit as st


model = pickle.load(open('model_linear.pkl', 'rb'))
# 398	396	9.4	25.2	25.4 = 269

def predict(IP_VoltageVoltageP2P,OP_VoltageP2N,OP_Current,Room_Temp,Battery_temp):

    df=pd.read_csv('ups.csv')
    df=df.drop(['Battery Voltage (V)','Unnamed: 0'],axis=1)
    num_arr=pd.DataFrame(np.array([[IP_VoltageVoltageP2P,OP_VoltageP2N,OP_Current,Room_Temp,Battery_temp]]),columns=df.columns)
    df_5=pd.concat([df,num_arr],axis=0)
    df_5.reset_index(drop=True,inplace=True)
    mms = MinMaxScaler()
    df_mms = pd.DataFrame(mms.fit_transform(df_5), columns = df_5.columns)
    df_last=df_mms.iloc[-1,:]
    my_prediction = model.predict(np.array([df_last]))

    return np.around(my_prediction,2)[0].astype('str')

def UI():
    st.write('#### Prediction for UPS')
    col1,col2,col3,col4 = st.columns([3,3,3,3])
    with col1:
        IP_Voltage = st.text_input('IP_VoltageVoltageP2P',398)
        Room_Temp = st.text_input('Room_Temp',25.2)
    with col2:
        OP_Voltage = st.text_input('OP_VoltageP2N',396)
        Battery_temp = st.text_input('Battery_temp',25.4)
    col1,col2,col3 = st.columns([6,3,3])
    OP_Current = col1.text_input('OP_Current',9.4)
    if col1.button('Predict'):
        Prediction = predict(IP_Voltage,OP_Voltage,OP_Current,Room_Temp,Battery_temp)
        print(Prediction)
        col1.write('#### Battery Voltage: ' + Prediction)

