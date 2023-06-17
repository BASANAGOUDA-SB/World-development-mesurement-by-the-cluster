import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_csv("clean_data.csv")
df = df.drop(['Number of Records'], axis=1)

def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)

df_norm = norm_func(df.iloc[:,:22])

clusters_new = KMeans(n_clusters=3, random_state=12)
cluster_model = clusters_new.fit(df_norm)

df['clusterid_new'] = clusters_new.labels_

def user_input_features():
    BR = st.sidebar.number_input("Insert the Birth Rate", min_value=0.0, step=0.001, max_value=1.0)
    BTR = st.sidebar.number_input("Insert the Business Tax Rate", min_value=0, step=1, max_value=350)
    CO2 = st.sidebar.number_input("Insert the CO2 Emissions")
    DSB = st.sidebar.number_input("Insert the Days to start Business", min_value=0, step=1, max_value=700)
    ENERGY = st.sidebar.number_input("Insert Energy Usage")
    GDP = st.sidebar.number_input("Insert GDP")
    HEP = st.sidebar.number_input("Insert Health Exp % GDP", min_value=0.0, step=0.001, max_value=1.0)
    HEC = st.sidebar.number_input("Insert Health Exp/Capita", min_value=0, step=1, max_value=10000)
    HDT = st.sidebar.number_input("Insert Hours to do Tax", min_value=0, step=1, max_value=3000)
    IFR = st.sidebar.number_input("Insert the Infant Mortality Rate", min_value=0.0, step=0.001, max_value=1.0)
    IU = st.sidebar.number_input("Insert the Internet Usage", min_value=0.0, step=0.001, max_value=1.0)
    LI = st.sidebar.number_input("Insert the Lending Interest", min_value=0.0, step=0.001, max_value=5.0)
    LEF = st.sidebar.number_input("Insert Life Expectancy Female", min_value=0, step=1, max_value=100)
    LEM = st.sidebar.number_input("Insert Life Expectancy Male", min_value=0, step=1, max_value=100)
    MPU = st.sidebar.number_input("Insert the Mobile Phone Usage", min_value=0.0, step=0.001, max_value=5.0)
    P014 = st.sidebar.number_input("Insert the Population 0-14", min_value=0.0, step=0.001, max_value=1.0)
    P1564 = st.sidebar.number_input("Insert the Population 15-64", min_value=0.0, step=0.001, max_value=1.0)
    P65 = st.sidebar.number_input("Insert the Population 65+", min_value=0.0, step=0.001, max_value=1.0)
    POT = st.sidebar.number_input("Insert Population Total")
    POU = st.sidebar.number_input("Insert the Population Urban", min_value=0.0, step=0.001, max_value=1.0)
    TIB = st.sidebar.number_input("Insert Tourism Inbound")
    TOB = st.sidebar.number_input("Insert Tourism Outbound")

    data = {'Birth Rate':BR,
            'Business Tax Rate':BTR,
            'CO2 Emissions':CO2,
            'Days to Start Business':DSB,
            'Energy Usage':ENERGY,
            'GDP': GDP,
            'Health Exp % GDP': HEP,
            'Health Exp/Capita': HEC,
            'Hours to do Tax': HDT,
            'Infant Mortality Rate': IFR,
            'Internet Usage': IU,
            'Lending Interest': LI,
            'Life Expectancy Female': LEF,
            'Life Expectancy Male': LEM,
            'Mobile Phone Usage': MPU,
            'Population 0-14': P014,
            'Population 15-64': P1564,
            'Population 65+': P65,
            'Population Total': POT,
            'Population Urban': POU,
            'Tourism Inbound': TIB,
            'Tourism Outbound': TOB}
    features = pd.DataFrame(data,index = [0])
    return features


df_p = user_input_features()
st.header('GOLBAL DEVELOPMENT MEASURMENTS')
st.subheader('User Input parameters')
st.write(df)

prediction = cluster_model.predict(df_p)

st.subheader('Prediction')
st.write(prediction)