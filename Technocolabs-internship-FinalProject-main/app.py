# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 10:50:23 2020

@author: prakhar
"""
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.write("""
# Online Shoppers Intention Prediction App
This app predicts the **Revenue** for the user based on his/her search history

""")
st.sidebar.header('User Input Features')

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        
        
        Administrative= st.sidebar.slider('Administrative',0,27,13)
        Informational = st.sidebar.slider('Informational', 0,24,12)
        Product_related = st.sidebar.slider('Product Related', 1,705,357)
        ProductRelated_Duration = st.sidebar.slider('ProductRelated Duration', 0.00000,70000.00000,35000.00000)
        Page_values=st.sidebar.slider('Page Values',0.0,370.0,185.0)
        Special_Day=st.sidebar.selectbox('Special Day',('0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0'))
        Month = st.sidebar.selectbox('Month',('Jan','Feb','March','April','May','June','July','Aug','Sep','Oct','Nov','Dec'))
        Visitor_Type = st.sidebar.selectbox('Visitor Type',('Returning Visitor','New Visitor','Other'))
        Weekend=st.sidebar.selectbox('Weekend',('1','0'))
        data = {
                'Administrative': Administrative,
                'Informational': Informational,
                'Product related':Product_related,
                'ProductRelated Duration': ProductRelated_Duration,
                'Page Values':Page_values,
                'Special Day':Special_Day,
                'Month': Month,
                'Visitor Type': Visitor_Type,
                'Weekend':Weekend
                }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()
    
    
    
# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
shoppers_raw = pd.read_csv('online_shoppers_intention.csv')
shoppers = shoppers_raw.drop(columns=["Region","TrafficType","OperatingSystems","Administrative_Duration","Informational_Duration","Browser","BounceRates","ExitRates","Revenue"],axis=1,inplace=True)
df = pd.concat([input_df,shoppers],axis=0)


 
df["Weekend"]=df["Weekend"].astype(int) 


#df = pd.get_dummies(df, columns=[ "Month","Visitor Type"], drop_first=True)

# Encoding of ordinal features

encode = [ "Month","Visitor Type"]
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('shoppers.pkl', 'rb'))


prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
revenue_value= np.array(['False','True'])
st.write(revenue_value[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
