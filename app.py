import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.preprocessing import LabelEncoder
from pandas.api.types import is_numeric_dtype

st.set_page_config(page_title='Hit Song Predictor', page_icon="🎵", layout="wide")
st.title('🎵 Hit Song Predictor')

DATA_PATH = 'data/spotify.csv'


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


try:
    df_init = load_data(DATA_PATH)
    st.success(f'✅ Dataset incarcat din: {DATA_PATH}')
except Exception as e:
    st.error('❌ Nu se poate incarca datasetul. Verifica path-ul!')
    st.exception(e)
    st.stop()

st.subheader('Preview dataset: ')
st.write(f'Shape: {df_init.shape}')
st.dataframe(df_init.head(20))

st.divider()

# analiza valorilor lipsa
st.subheader('Analiza valorilor lipsa din setul de date')

total = df_init.isnull().sum().sort_values(ascending=False)
percent = (df_init.isnull().sum() * 100 / df_init.isnull().count()).sort_values(ascending=False)

missing = pd.concat([total, percent], axis=1, keys=['Total', 'Procent'])
st.dataframe(missing)

st.divider()

# tratare valori lipsa
st.subheader('Tratarea valorilor lipsa din setul de date')


def fill_nan(df: pd.DataFrame):
    for c in df.columns:
        if df[c].isna().any():
            if is_numeric_dtype(df[c]):
                df.fillna({c: df[c].mean()}, inplace=True)
            else:
                df.fillna({c: df[c].mode().iloc[0]}, inplace=True)


df = df_init.copy()
fill_nan(df)
st.write(f'Valori lipsa ramase: {df.isnull().sum().sum()}')

st.divider()

# tratarea valorilor extreme
st.subheader('Tratarea valorilor extreme din setul de date')


def remove_outliers(data, col):
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return data[(data[col] >= lower) & (data[col] <= upper)]


if 'tempo' in df.columns:
    nr_initial = len(df)
    df = remove_outliers(df, 'tempo')
    nr_final = len(df)

    st.write(f'Numarul de observatii inainte de eliminarea valorilor outliers: {nr_initial}')
    st.write(f'Numar observatii dupa eliminarea valorilor outliers: {nr_final}')
else:
    st.warning("Coloana 'tempo' nu exista in dataset")

st.divider()

# statistici
st.subheader('Statistici')

st.write('Statistici descriptive pentru coloanele numerice: ')
st.dataframe(df.describe())

if 'popularity' in df.columns:
    df['hit'] = (df['popularity'] > 70).astype(int)
else:
    st.error("Coloana 'popularity' nu exista in dataset")
    st.stop()

st.divider()

st.subheader('Gruparea si agregarea datelor')

hit_stats = df.groupby('hit').agg({
    'popularity': ['mean', 'count', 'max', 'min'],
    'danceability': 'mean',
    'energy': 'mean',
    'tempo': 'mean'
})

df['hit_label'] = df['hit'].map({0: 'non-HIT', 1: 'HIT'})

col1, col2, col3 = st.columns(3)
col1.metric('Total melodii: ', len(df))
col2.metric('Numar HIT-uri: ', int((df['hit'] == 1).sum()))
col3.metric('Numar non-HIT-uri: ', int((df['hit'] == 0).sum()))

st.dataframe(hit_stats)

if 'track_genre' in df.columns:
    genre_stats = df.groupby('track_genre').agg({
        'popularity': ['mean', 'count'],
        'duration_ms': 'mean'
    }).sort_values(('popularity', 'mean'), ascending=False).head(10)

    st.write('Top 10 genuri muzicale dupa popularitatea medie:')
    st.dataframe(genre_stats)

st.divider()

st.subheader('Utilizare functiilor de grup')


def amplitudine(x):
    return x.max() - x.min()


result = df.groupby('hit')['energy'].apply(amplitudine)

st.write("Amplitudinea valorilor 'energy' pentru fiecare grup hit/non-hit")
st.dataframe(result)

st.divider()

st.subheader('Reprezentari grafice')

st.write('Numar melodii HIT vs non-HIT: ')
st.bar_chart(df['hit_label'].value_counts())

if 'track_genre' in df.columns:
    top_genuri = df['track_genre'].value_counts().head(10)
    st.write('Top 10 genuri dupa numarul de melodii')
    st.write(top_genuri)

st.divider()

st.subheader('Codificarea datelor')

if 'track_genre' in df.columns:
    encoder = LabelEncoder()
    df['track_genre_encoded'] = encoder.fit_transform(df['track_genre'])

    st.write("Exemplu de codificare pentru coloana 'track_genre': ")
    st.dataframe(df[['track_genre', 'track_genre_encoded']].drop_duplicates().head(10))
else:
    st.warning("Coloana 'track_genre' nu exista in dataset")
