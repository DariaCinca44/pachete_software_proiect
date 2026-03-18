import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import geodatasets
from sklearn.preprocessing import LabelEncoder,StandardScaler, MinMaxScaler
from pandas.api.types import is_numeric_dtype
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import statsmodels.api as sm
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

st.divider()
st.subheader('Selectbox alegere genului')
if 'track_genre' in df.columns:
    gen_selectat = st.sidebar.selectbox(
        'Alege un gen muzical',
        ['Toate'] + sorted(df['track_genre'].unique().tolist())
    )

    if gen_selectat != 'Toate':
        df_filtered = df[df['track_genre'] == gen_selectat]
    else:
        df_filtered = df.copy()

    st.subheader('Date dupa selectarea genului')
    st.dataframe(df_filtered.head(20))

st.divider()
st.subheader('Corelatia dintre variabilele numerice')

corr = df.select_dtypes(include=[np.number]).corr()
st.dataframe(corr)
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(corr, aspect='auto')
ax.set_xticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=90)
ax.set_yticks(range(len(corr.columns)))
ax.set_yticklabels(corr.columns)
plt.colorbar(im)
plt.tight_layout()
st.pyplot(fig)

st.divider()
st.subheader('Descarcare date prelucrate')

csv = df.to_csv(index=False).encode('utf-8')

st.download_button(
    label='Descarca datasetul prelucrat',
    data=csv,
    file_name='spotify_prelucrat.csv',
    mime='text/csv'
)

st.divider()
st.subheader('Metode de scalare')

coloane_scalare = ['danceability', 'energy', 'tempo', 'valence']
coloane_scalare = [c for c in coloane_scalare if c in df.columns]

if len(coloane_scalare) > 0:
    scaler_standard = StandardScaler()
    scaler_minmax = MinMaxScaler()

    df_standardizat = pd.DataFrame(
        scaler_standard.fit_transform(df[coloane_scalare]),
        columns=[c + '_std' for c in coloane_scalare]
    )

    df_minmax = pd.DataFrame(
        scaler_minmax.fit_transform(df[coloane_scalare]),
        columns=[c + '_minmax' for c in coloane_scalare]
    )

    st.write('Exemplu StandardScaler')
    st.dataframe(df_standardizat.head(10))

    st.write('Exemplu MinMaxScaler')
    st.dataframe(df_minmax.head(10))

st.divider()
st.subheader('Scikit-learn: clusterizare KMeans')

cluster_features = ['danceability', 'energy', 'valence', 'tempo']
cluster_features = [c for c in cluster_features if c in df.columns]

if len(cluster_features) >= 2:
    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(df[cluster_features])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_cluster)

    st.write('Numar melodii pe cluster')
    st.dataframe(df['cluster'].value_counts().sort_index())

st.divider()
st.subheader('Scikit-learn: regresie logistica')

model_features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'tempo']
model_features = [c for c in model_features if c in df.columns]

if len(model_features) > 0:
    X = df[model_features]
    y = df['hit']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    st.write(f'Acuratete: {accuracy_score(y_test, y_pred):.4f}')
    st.dataframe(pd.DataFrame(confusion_matrix(y_test, y_pred)))
    st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())

st.divider()
st.subheader('Statsmodels: regresie multipla')

ols_features = ['danceability', 'energy', 'tempo', 'valence']
ols_features = [c for c in ols_features if c in df.columns]

if len(ols_features) > 0 and 'popularity' in df.columns:
    X_ols = df[ols_features]
    X_ols = sm.add_constant(X_ols)
    y_ols = df['popularity']

    model_ols = sm.OLS(y_ols, X_ols).fit()

    st.write(f'R-squared: {model_ols.rsquared:.4f}')
    st.text(model_ols.summary().as_text())

st.divider()
st.subheader('Utilizarea pachetului geopandas')

world = gpd.read_file(geodatasets.get_path("geoda.chicago_commpop"))
st.dataframe(world.head(10))

fig, ax = plt.subplots(figsize=(10, 6))
world.plot(ax=ax)
st.pyplot(fig)
