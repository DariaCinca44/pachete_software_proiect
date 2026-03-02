import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

st.set_page_config(page_title='Hit Song Predictor', page_icon="🎵", layout= "wide")
st.title('🎵 Hit Song Predictor')

DATA_PATH= 'data/spotify.csv'

@st.cache_data
def load_data(path: str)-> pd.DataFrame:
    return pd.read_csv(path)

try:
    df_init= load_data(DATA_PATH)
    st.success(f'✅ Dataset incarcat din: {DATA_PATH}')
except Exception as e:
    st.error('❌ Nu se poate incarca datasetul. Verifica path-ul!')
    st.exception(e)
    st.stop()

st.subheader('Preview dataset: ')
st.write(f'Shape: {df_init.shape}')
st.dataframe(df_init)
