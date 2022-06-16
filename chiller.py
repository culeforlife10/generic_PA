from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from datetime import datetime,timedelta
pd.set_option('display.width', 5000)

def color_survived(val):
    #color = 'green' if val=='Good' else 'red'
    if val=='Good':
        color='green'
    elif val=='Average' or val=='Above Average':
        color='yellow'
    else:
        color='red'
    return f'background-color: {color}'

def predict(end_date):
    new_df=pd.read_excel('chiller april-march.xlsx')
    new_df1=new_df.copy()
    # end_date=datetime.strptime('03/01/2022',"%m/%d/%Y")#input shld be fed here in months,days,years format
    end_date = end_date - timedelta(days=3)
    date1=pd.date_range(start=end_date+timedelta(days=3),end='3/31/2022')
    final_df=pd.DataFrame()
    final_df1=pd.DataFrame()
    final_df['date']=date1
    final_df1['date']=date1
    new_df.set_index('Timestamp',inplace=True)
    train=new_df[:end_date]
    test=new_df[end_date:]
    raw_seq=new_df.values
    mms=MinMaxScaler()
    scaled=mms.fit_transform(raw_seq)
    X=[]
    y=[]
    steps=2
    for i in range(len(scaled)):
        end_ix = i + steps
        if end_ix > len(scaled)-1:
            break
        seq_x, seq_y = scaled[i:end_ix], scaled[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    X=np.array(X)
    y=np.array(y)
    val=len(train)
    xtest=X[val:]
    ytest=y[val:]
    val1=len(ytest)
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    model=load_model('model_lstm.h5')
    ypred=[]
    for i in range(len(xtest)):
        x_input=xtest[i]
        x_input = x_input.reshape((1, steps, n_features))
        yhat = model.predict(x_input, verbose=0)
        ypred.append(yhat)
    ypred=np.array(ypred)
    ypred=ypred.reshape(val1,1)
    ytest_scaled=mms.inverse_transform(ytest)
    ypred_scaled=mms.inverse_transform(ypred)
    final_df['kwh']=ypred_scaled   # this shld be shown as output. This is our predicted output
    final_df1['kwh']=ytest_scaled
    ydf = final_df.set_index('date')
    final_df1.set_index('date',inplace=True)
  
    remarks=[]

    
    for i in final_df['kwh']:
        if i>=0 and i<=5000:
            i='Good'
            remarks.append(i)
            
            

        elif i>=5000 and i<= 16000:

            i='Average'
            remarks.append(i)
           

        elif i>=16000 and i<= 20000:
            i='Above Average'
            remarks.append(i)
            

        else:
            i='High Consumption'
            remarks.append(i)
            

        

    final_df['Remarks']=remarks
    
    plt.plot(ydf)
    
    plt.plot(final_df1)
    plt.plot()
    plt.xticks(rotation=40)
    plt.grid(True,axis='y')
    plt.xlabel("Dates")
    plt.ylabel("Consumption")
    plt.savefig('testvspred.png',bbox_inches = 'tight')
    final_df['date']=final_df['date'].dt.date
    
    return final_df.style.applymap(color_survived, subset=['Remarks'])

def UI():
    st.write('#### Prediction for Chiller Machines')
    #col1,col2,col3 = st.columns(3)
    end_date = st.date_input('Upto what date would you like to train the model on ?',datetime.strptime('03/01/2022',"%m/%d/%Y"),
    min_value=datetime.strptime('06/30/2021',"%m/%d/%Y"),max_value= datetime.strptime('03/31/2022',"%m/%d/%Y"))
    col1=st.columns(5)
    if st.button('Predict'):
        col1.write(predict(end_date),1000,1000)
        graph = Image.open('testvspred.png')
        st.image(graph)
    




