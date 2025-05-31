# === IMPORTAR LIBRERÍAS ===
import os
import pandas as pd
import numpy as np
import io
!pip install nltk
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download("vader_lexicon")
nltk.download("stopwords")
from google.cloud import bigquery, storage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
!pip install tensorflow
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from collections import Counter
import re  # Añadido para tokenización en regex

# === VARIABLES PARA CARGAR DATOS ===
PROJECT_ID = "lyrics-success-452515"
DATASET_ID = "lyrics_success_dataset"
GCS_BUCKET = "lyrics-success-bucket0"

# === CLIENTES GOOGLE CLOUD ===
storage_client = storage.Client()
bigquery_client = bigquery.Client()

# === FUNCIÓN PARA CARGAR CSVs DESDE GCS ===
def load_csv_from_gcs(bucket_name, file_name):
    """Convertir CSVs de GCS en DataFrames de Pandas."""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    content = blob.download_as_text()
    return pd.read_csv(io.StringIO(content))

# === FUNCIÓN PARA EXPORTAR DATOS A BIG QUERY ===
def load_dataframe_to_bigquery(df, table_name):
    """Load a Pandas DataFrame directly into BigQuery."""
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{table_name}"

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        autodetect=True  
    )

    job = bigquery_client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()
    print(f"Tabla {table_name} exportada a Big Query con éxito")

# === CARGAR DATOS ===
print("Cargando datasets desde Google Cloud Storage...")
predictions_df = load_csv_from_gcs(GCS_BUCKET, "predictions_dataset.csv")  # Entrenamiento & Predicciones ML
analytics_df = load_csv_from_gcs(GCS_BUCKET, "analytics_dataset_14.csv")  # Visualizaciones Looker
user_input_df = load_csv_from_gcs(GCS_BUCKET, "4_song_user_input.csv")  # Entrada de datos del usuario

# === LOOP PARA ELIMINAR COLUMNA INNECESARIA === 
for df in [predictions_df, analytics_df, user_input_df]:
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

# === INICIAR ANÁLISIS DE SENTIMIENTO ===
stop_words = set(nltk.corpus.stopwords.words('english'))
sia = SentimentIntensityAnalyzer()

def get_vader_sentiment(text):
    """Calcular sentimiento (compound) usando VADER."""
    scores = sia.polarity_scores(str(text))
    return float(scores['compound'])

# === APLICAR ANÁLISIS DE SENTIMIENTO A AMBOS CONJUNTOS DE DATOS ===
if "compound" not in analytics_df.columns:
    print("Aplicando análisis de sentimiento a analytics dataset...")
    analytics_df['compound'] = analytics_df['lyrics'].apply(get_vader_sentiment)

if "compound" not in predictions_df.columns:
    print("Aplicando análisis de sentimiento a predictions dataset...")
    predictions_df['compound'] = predictions_df['lyrics'].apply(get_vader_sentiment)

# === DATOS PARA VIEWS PIE CHART (Origen = Analytics Dataset) ===
views_pie_genre_year = (
    analytics_df.groupby(['year', 'tag'])['views']
    .sum()
    .reset_index()
)

# === DATOS PARA TOTAL VIEWS PER YEAR ===
views_total_year = (
    analytics_df.groupby('year')['views']
    .sum()
    .reset_index()
    .rename(columns={'views': 'total_views'})
)

# === 🟢 DATOS PARA COMPOUND RADAR CHART DATA (Sentimiento anual medio por género) ===
sent_avg_genre_year = (
    analytics_df.groupby(['year', 'tag'])['compound']
    .mean()
    .reset_index()
    .pivot(index='year', columns='tag', values='compound')
    .reset_index()
)

# Renombrar columnas
sent_avg_genre_year.columns = ['year'] + [f"{col}_avg_compound" for col in sent_avg_genre_year.columns[1:]]


# === DATOS PARA SENTIMENT AVERAGE PER YEAR (Sentimiento medio por año) ===
sent_avg_year = (
    analytics_df.groupby('year')['compound']
    .mean()
    .reset_index()
    .rename(columns={'compound': 'avg_compound_sentiment'})  # Renombrar columnas
)

# === DATOS PARA ALL-TIME RADAR CHART (Medias por género) ===
sent_radar_all_time = (
    analytics_df.groupby('tag')['compound']
    .mean()
    .reset_index()
    .pivot(columns='tag', values='compound')
)

# Renombrar columnas
sent_radar_all_time.columns = [f"{col}_avg_sent" for col in sent_radar_all_time.columns]

# Añadir columna estática, necesaria para la visualización
sent_radar_all_time['all_time'] = 'All Time'


# === DATOS PARA WORD TREEMAP ===
def preprocess_lyrics(text):
    words = re.findall(r'\b[a-zA-Z]+\b', str(text).lower())
    meaningful_words = [word for word in words if word not in stop_words and len(word) > 2]
    return meaningful_words

analytics_df['processed_lyrics'] = analytics_df['lyrics'].apply(preprocess_lyrics)

def get_top_words(group):
    all_words = [word for lyrics in group['processed_lyrics'] for word in lyrics]
    word_freq = Counter(all_words).most_common(25)
    return pd.DataFrame(word_freq, columns=['word', 'frequency'])

word_treemap = (
    analytics_df.groupby(['year', 'tag'])
    .apply(get_top_words)
    .reset_index()
)

# === PREPROCESADO PARA MODELOS ML ===
print("Encoding categorical data for model training...")
label_encoders = {}
for column in ['tag', 'artist']:
    le = LabelEncoder()
    predictions_df[column] = le.fit_transform(predictions_df[column].astype(str))
    label_encoders[column] = le

# Estandarizar 'year'
scaler = StandardScaler()
predictions_df[['year']] = scaler.fit_transform(predictions_df[['year']])

# Codificar user input para las predicciones
for column in ['tag', 'artist']:
    if column in label_encoders:
        le = label_encoders[column]
        user_input_df[column] = user_input_df[column].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

user_input_df[['year']] = scaler.transform(user_input_df[['year']])


# === ENTRENAR MODELO DE PREDICCIÓN DE SENTIMIENTOS (MODELO 1) ===
print("Training sentiment prediction model...")
X_sentiment = predictions_df['lyrics']
y_sentiment = predictions_df['compound']
X_train, X_test, y_train, y_test = train_test_split(X_sentiment, y_sentiment, test_size=0.2, random_state=42)

tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model1_nn = Sequential([
    Input(shape=(X_train_tfidf.shape[1],)),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='tanh')
])
model1_nn.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model1_nn.fit(X_train_tfidf, y_train, epochs=20, batch_size=16, validation_split=0.2, verbose=2)

# === EVALUACIÓN DEL MODELO 1 (Predicción de sentimiento) ===
# Predecir sobre el subgrupo de entrenamiento (test set)
predicted_sentiment = model1_nn.predict(X_test_tfidf)
# Calcular Mean Absolute Error (MAE)
mae1 = tf.keras.losses.MeanAbsoluteError()(y_test, predicted_sentiment).numpy()
# Calcular R-squared (R2 Score)
r2_1 = 1 - (np.sum((y_test - predicted_sentiment.squeeze())**2) / np.sum((y_test - np.mean(y_test))**2))
print(f"Model 1 Evaluation -> MAE: {mae1:.4f}, R2: {r2_1:.4f}")

# Predecir sentimiento para los datos del usuario
user_input_tfidf = tfidf.transform(user_input_df['lyrics'])
user_input_df['predicted_compound'] = model1_nn.predict(user_input_tfidf)
user_input_df['predicted_compound'] = user_input_df['predicted_compound'].clip(-1, 1)

# === ENTRENAR MODELO DE PREDICCIÓN DE VISITAS (MODELO 2) ===
print("Training views prediction model...")
X_views = predictions_df[['tag', 'artist', 'year', 'compound']]
y_views = predictions_df['views']

# Convertir a datos numéricos (compatibilidad)
X_views = X_views.apply(pd.to_numeric, errors='coerce')
y_views = y_views.apply(pd.to_numeric, errors='coerce')

# Buscar NaNs y tratar valores inexistentes
X_views.fillna(0, inplace=True)
y_views.fillna(0, inplace=True)

X_views_train, X_views_test, y_views_train, y_views_test = train_test_split(X_views, y_views, test_size=0.2, random_state=42)

model2_nn = Sequential([
    Input(shape=(X_views_train.shape[1],)),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])
model2_nn.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model2_nn.fit(X_views_train, y_views_train, epochs=50, batch_size=32, validation_split=0.2, verbose=2)

# === EVALUACIÓN MODELO 2 (Predicción de Views) ===

# Predecir sobre el subgrupo de entrenamiento (test set)
predicted_views = model2_nn.predict(X_views_test)

# Calcular Mean Absolute Error (MAE)
mae2 = tf.keras.losses.MeanAbsoluteError()(y_views_test, predicted_views).numpy()

# Calcular R-squared (R2 Score)
r2_2 = 1 - (np.sum((y_views_test - predicted_views.squeeze())**2) / np.sum((y_views_test - np.mean(y_views_test))**2))

print(f"Model 2 Evaluation -> MAE: {mae2:.2f}, R2: {r2_2:.2f}")

# === ANÁLISIS DE IMPORTANCIA DE FACTORES (FEATURE IMPORTANCE, MODELO 2) ===
# Extraer suma absoluta de pesos desde la primera capa (input layer)
feature_importance = np.abs(model2_nn.layers[0].get_weights()[0]).sum(axis=1)

# Calcular el peso total de todos los factores
total_weight = feature_importance.sum()

# Create DataFrame with both absolute and relative importance
feature_importance_df = pd.DataFrame({
    'Feature': X_views_train.columns,  # Nombres de los factores
    'Absolute_Importance': feature_importance,
    'Relative_Importance': (feature_importance / total_weight) * 100  # Convertir a porcentaje
})

# Ordenar por importancia
feature_importance_df = feature_importance_df.sort_values(by="Absolute_Importance", ascending=False)

# Mostrar importancia de factores por peso total
print(f"Total Feature Weight: {total_weight:.4f}")
print("Feature Importance for Model 2:")
print(feature_importance_df)

# Predecir visitas para canciones del usuario
user_input_views = user_input_df[['tag', 'artist', 'year', 'predicted_compound']]
user_input_df['predicted_views'] = model2_nn.predict(user_input_views)

# === EXPORTAR A BIG QUERY ===
print("Exporting structured data to BigQuery...")

load_dataframe_to_bigquery(user_input_df[['title', 'predicted_compound', 'predicted_views']], "user_input_predictions_genre0")
load_dataframe_to_bigquery(views_pie_genre_year[['year', 'tag', 'views']], "views_pie_genre_year0")
load_dataframe_to_bigquery(views_total_year[['year', 'total_views']], "views_total_year0")
load_dataframe_to_bigquery(sent_avg_genre_year, "sent_avg_genre_year0")
load_dataframe_to_bigquery(sent_avg_year[['year', 'avg_compound_sentiment']], "sent_avg_year0")
load_dataframe_to_bigquery(sent_radar_all_time, "sent_radar_all_time0")
load_dataframe_to_bigquery(word_treemap[['year', 'tag', 'word', 'frequency']], "word_treemap0")
load_dataframe_to_bigquery(feature_importance_df, "feature_importance_model0")

print("Todas las tablas estructuradas y exportadas a Big Query con éxito")


"""Requirement already satisfied: nltk in /opt/conda/lib/python3.10/site-packages (3.9.1)
Requirement already satisfied: click in /opt/conda/lib/python3.10/site-packages (from nltk) (8.1.8)
Requirement already satisfied: joblib in /opt/conda/lib/python3.10/site-packages (from nltk) (1.4.2)
Requirement already satisfied: regex>=2021.8.3 in /opt/conda/lib/python3.10/site-packages (from nltk) (2024.11.6)
Requirement already satisfied: tqdm in /opt/conda/lib/python3.10/site-packages (from nltk) (4.67.1)

[notice] A new release of pip is available: 25.0 -> 25.0.1
[notice] To update, run: pip install --upgrade pip
[nltk_data] Downloading package vader_lexicon to
[nltk_data]     /home/jupyter/nltk_data...
[nltk_data]   Package vader_lexicon is already up-to-date!
[nltk_data] Downloading package stopwords to
[nltk_data]     /home/jupyter/nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
Requirement already satisfied: tensorflow in /opt/conda/lib/python3.10/site-packages (2.18.0)
Requirement already satisfied: absl-py>=1.0.0 in /opt/conda/lib/python3.10/site-packages (from tensorflow) (2.1.0)
Requirement already satisfied: astunparse>=1.6.0 in /opt/conda/lib/python3.10/site-packages (from tensorflow) (1.6.3)
Requirement already satisfied: flatbuffers>=24.3.25 in /opt/conda/lib/python3.10/site-packages (from tensorflow) (25.2.10)
Requirement already satisfied: gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 in /opt/conda/lib/python3.10/site-packages (from tensorflow) (0.6.0)
Requirement already satisfied: google-pasta>=0.1.1 in /opt/conda/lib/python3.10/site-packages (from tensorflow) (0.2.0)
Requirement already satisfied: libclang>=13.0.0 in /opt/conda/lib/python3.10/site-packages (from tensorflow) (18.1.1)
Requirement already satisfied: opt-einsum>=2.3.2 in /opt/conda/lib/python3.10/site-packages (from tensorflow) (3.4.0)
Requirement already satisfied: packaging in /opt/conda/lib/python3.10/site-packages (from tensorflow) (24.2)
Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<6.0.0dev,>=3.20.3 in /opt/conda/lib/python3.10/site-packages (from tensorflow) (3.20.3)
Requirement already satisfied: requests<3,>=2.21.0 in /opt/conda/lib/python3.10/site-packages (from tensorflow) (2.32.3)
Requirement already satisfied: setuptools in /opt/conda/lib/python3.10/site-packages (from tensorflow) (75.8.0)
Requirement already satisfied: six>=1.12.0 in /opt/conda/lib/python3.10/site-packages (from tensorflow) (1.17.0)
Requirement already satisfied: termcolor>=1.1.0 in /opt/conda/lib/python3.10/site-packages (from tensorflow) (2.5.0)
Requirement already satisfied: typing-extensions>=3.6.6 in /opt/conda/lib/python3.10/site-packages (from tensorflow) (4.12.2)
Requirement already satisfied: wrapt>=1.11.0 in /opt/conda/lib/python3.10/site-packages (from tensorflow) (1.17.2)
Requirement already satisfied: grpcio<2.0,>=1.24.3 in /opt/conda/lib/python3.10/site-packages (from tensorflow) (1.70.0)
Requirement already satisfied: tensorboard<2.19,>=2.18 in /opt/conda/lib/python3.10/site-packages (from tensorflow) (2.18.0)
Requirement already satisfied: keras>=3.5.0 in /opt/conda/lib/python3.10/site-packages (from tensorflow) (3.8.0)
Requirement already satisfied: numpy<2.1.0,>=1.26.0 in /opt/conda/lib/python3.10/site-packages (from tensorflow) (1.26.4)
Requirement already satisfied: h5py>=3.11.0 in /opt/conda/lib/python3.10/site-packages (from tensorflow) (3.13.0)
Requirement already satisfied: ml-dtypes<0.5.0,>=0.4.0 in /opt/conda/lib/python3.10/site-packages (from tensorflow) (0.4.1)
Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in /opt/conda/lib/python3.10/site-packages (from tensorflow) (0.37.1)
Requirement already satisfied: wheel<1.0,>=0.23.0 in /opt/conda/lib/python3.10/site-packages (from astunparse>=1.6.0->tensorflow) (0.45.1)
Requirement already satisfied: rich in /opt/conda/lib/python3.10/site-packages (from keras>=3.5.0->tensorflow) (13.9.4)
Requirement already satisfied: namex in /opt/conda/lib/python3.10/site-packages (from keras>=3.5.0->tensorflow) (0.0.8)
Requirement already satisfied: optree in /opt/conda/lib/python3.10/site-packages (from keras>=3.5.0->tensorflow) (0.14.1)
Requirement already satisfied: charset_normalizer<4,>=2 in /opt/conda/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorflow) (3.4.1)
Requirement already satisfied: idna<4,>=2.5 in /opt/conda/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorflow) (3.10)
Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/conda/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorflow) (1.26.20)
Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.10/site-packages (from requests<3,>=2.21.0->tensorflow) (2024.12.14)
Requirement already satisfied: markdown>=2.6.8 in /opt/conda/lib/python3.10/site-packages (from tensorboard<2.19,>=2.18->tensorflow) (3.7)
Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in /opt/conda/lib/python3.10/site-packages (from tensorboard<2.19,>=2.18->tensorflow) (0.7.2)
Requirement already satisfied: werkzeug>=1.0.1 in /opt/conda/lib/python3.10/site-packages (from tensorboard<2.19,>=2.18->tensorflow) (3.1.3)
Requirement already satisfied: MarkupSafe>=2.1.1 in /opt/conda/lib/python3.10/site-packages (from werkzeug>=1.0.1->tensorboard<2.19,>=2.18->tensorflow) (3.0.2)
Requirement already satisfied: markdown-it-py>=2.2.0 in /opt/conda/lib/python3.10/site-packages (from rich->keras>=3.5.0->tensorflow) (3.0.0)
Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /opt/conda/lib/python3.10/site-packages (from rich->keras>=3.5.0->tensorflow) (2.19.1)
Requirement already satisfied: mdurl~=0.1 in /opt/conda/lib/python3.10/site-packages (from markdown-it-py>=2.2.0->rich->keras>=3.5.0->tensorflow) (0.1.2)

[notice] A new release of pip is available: 25.0 -> 25.0.1
[notice] To update, run: pip install --upgrade pip
2025-03-08 19:23:04.018895: I external/local_xla/xla/tsl/cuda/cudart_stub.cc:32] Could not find cuda drivers on your machine, GPU will not be used.
2025-03-08 19:23:04.048998: I external/local_xla/xla/tsl/cuda/cudart_stub.cc:32] Could not find cuda drivers on your machine, GPU will not be used.
2025-03-08 19:23:04.089982: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1741461784.131204   57484 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1741461784.144445   57484 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2025-03-08 19:23:04.220642: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
Cargando datasets desde Google Cloud Storage...
Aplicando análisis de sentimiento a analytics dataset...
Aplicando análisis de sentimiento a predictions dataset...
/var/tmp/ipykernel_57484/2934630113.py:148: DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
  .apply(get_top_words)
Encoding categorical data for model training...
Training sentiment prediction model...
2025-03-08 19:40:03.566834: E external/local_xla/xla/stream_executor/cuda/cuda_driver.cc:152] failed call to cuInit: INTERNAL: CUDA error: Failed call to cuInit: UNKNOWN ERROR (303)
Epoch 1/20
10000/10000 - 69s - 7ms/step - loss: 0.3133 - mae: 0.3848 - val_loss: 0.2709 - val_mae: 0.3522
Epoch 2/20
10000/10000 - 66s - 7ms/step - loss: 0.2526 - mae: 0.3212 - val_loss: 0.2627 - val_mae: 0.3394
Epoch 3/20
10000/10000 - 66s - 7ms/step - loss: 0.2153 - mae: 0.2842 - val_loss: 0.2612 - val_mae: 0.3173
Epoch 4/20
10000/10000 - 67s - 7ms/step - loss: 0.1779 - mae: 0.2471 - val_loss: 0.2721 - val_mae: 0.3074
Epoch 5/20
10000/10000 - 67s - 7ms/step - loss: 0.1509 - mae: 0.2206 - val_loss: 0.2776 - val_mae: 0.3136
Epoch 6/20
10000/10000 - 64s - 6ms/step - loss: 0.1316 - mae: 0.2021 - val_loss: 0.2823 - val_mae: 0.3134
Epoch 7/20
10000/10000 - 63s - 6ms/step - loss: 0.1192 - mae: 0.1902 - val_loss: 0.2919 - val_mae: 0.3191
Epoch 8/20
10000/10000 - 64s - 6ms/step - loss: 0.1075 - mae: 0.1788 - val_loss: 0.2960 - val_mae: 0.3072
Epoch 9/20
10000/10000 - 65s - 6ms/step - loss: 0.0990 - mae: 0.1705 - val_loss: 0.2975 - val_mae: 0.3195
Epoch 10/20
10000/10000 - 64s - 6ms/step - loss: 0.0926 - mae: 0.1641 - val_loss: 0.2997 - val_mae: 0.3152
Epoch 11/20
10000/10000 - 64s - 6ms/step - loss: 0.0864 - mae: 0.1580 - val_loss: 0.3026 - val_mae: 0.3184
Epoch 12/20
10000/10000 - 63s - 6ms/step - loss: 0.0819 - mae: 0.1531 - val_loss: 0.3003 - val_mae: 0.3104
Epoch 13/20
10000/10000 - 65s - 6ms/step - loss: 0.0772 - mae: 0.1483 - val_loss: 0.3084 - val_mae: 0.3096
Epoch 14/20
10000/10000 - 64s - 6ms/step - loss: 0.0738 - mae: 0.1447 - val_loss: 0.3039 - val_mae: 0.3140
Epoch 15/20
10000/10000 - 63s - 6ms/step - loss: 0.0705 - mae: 0.1413 - val_loss: 0.3028 - val_mae: 0.3107
Epoch 16/20
10000/10000 - 64s - 6ms/step - loss: 0.0689 - mae: 0.1390 - val_loss: 0.3077 - val_mae: 0.3127
Epoch 17/20
10000/10000 - 63s - 6ms/step - loss: 0.0651 - mae: 0.1350 - val_loss: 0.3121 - val_mae: 0.3129
Epoch 18/20
10000/10000 - 63s - 6ms/step - loss: 0.0629 - mae: 0.1323 - val_loss: 0.3110 - val_mae: 0.3155
Epoch 19/20
10000/10000 - 65s - 7ms/step - loss: 0.0615 - mae: 0.1306 - val_loss: 0.3127 - val_mae: 0.3125
Epoch 20/20
10000/10000 - 64s - 6ms/step - loss: 0.0598 - mae: 0.1285 - val_loss: 0.3090 - val_mae: 0.3145
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 5s 3ms/step
Model 1 Evaluation -> MAE: 0.3173, R2: 0.5889
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 72ms/step
Training views prediction model...
Epoch 1/50
5000/5000 - 13s - 3ms/step - loss: 20870562.0000 - mae: 1716.1500 - val_loss: 20233764.0000 - val_mae: 1577.6866
Epoch 2/50
5000/5000 - 13s - 3ms/step - loss: 20742512.0000 - mae: 1696.4368 - val_loss: 20270682.0000 - val_mae: 1539.3168
Epoch 3/50
5000/5000 - 13s - 3ms/step - loss: 20738780.0000 - mae: 1695.0964 - val_loss: 20178056.0000 - val_mae: 1664.1857
Epoch 4/50
5000/5000 - 13s - 3ms/step - loss: 20733622.0000 - mae: 1695.2220 - val_loss: 20172468.0000 - val_mae: 1680.5474
Epoch 5/50
5000/5000 - 13s - 3ms/step - loss: 20726454.0000 - mae: 1696.3293 - val_loss: 20168626.0000 - val_mae: 1693.3186
Epoch 6/50
5000/5000 - 12s - 2ms/step - loss: 20725620.0000 - mae: 1696.7321 - val_loss: 20194200.0000 - val_mae: 1616.6403
Epoch 7/50
5000/5000 - 12s - 2ms/step - loss: 20714510.0000 - mae: 1700.5056 - val_loss: 20169502.0000 - val_mae: 1671.2368
Epoch 8/50
5000/5000 - 13s - 3ms/step - loss: 20712784.0000 - mae: 1699.9467 - val_loss: 20163102.0000 - val_mae: 1725.1892
Epoch 9/50
5000/5000 - 15s - 3ms/step - loss: 20709614.0000 - mae: 1700.9316 - val_loss: 20160578.0000 - val_mae: 1704.2526
Epoch 10/50
5000/5000 - 13s - 3ms/step - loss: 20702266.0000 - mae: 1702.1228 - val_loss: 20181686.0000 - val_mae: 1625.5762
Epoch 11/50
5000/5000 - 13s - 3ms/step - loss: 20700066.0000 - mae: 1703.4642 - val_loss: 20187910.0000 - val_mae: 1611.3011
Epoch 12/50
5000/5000 - 13s - 3ms/step - loss: 20698406.0000 - mae: 1702.6296 - val_loss: 20158032.0000 - val_mae: 1671.1399
Epoch 13/50
5000/5000 - 13s - 3ms/step - loss: 20694398.0000 - mae: 1704.7992 - val_loss: 20213050.0000 - val_mae: 1572.0753
Epoch 14/50
5000/5000 - 13s - 3ms/step - loss: 20695882.0000 - mae: 1705.2214 - val_loss: 20205398.0000 - val_mae: 1580.5105
Epoch 15/50
5000/5000 - 13s - 3ms/step - loss: 20691566.0000 - mae: 1704.3292 - val_loss: 20143724.0000 - val_mae: 1724.9154
Epoch 16/50
5000/5000 - 13s - 3ms/step - loss: 20695510.0000 - mae: 1705.2874 - val_loss: 20267500.0000 - val_mae: 1522.0698
Epoch 17/50
5000/5000 - 14s - 3ms/step - loss: 20681274.0000 - mae: 1707.6522 - val_loss: 20163126.0000 - val_mae: 1619.8370
Epoch 18/50
5000/5000 - 14s - 3ms/step - loss: 20681978.0000 - mae: 1708.0494 - val_loss: 20276528.0000 - val_mae: 1511.1951
Epoch 19/50
5000/5000 - 13s - 3ms/step - loss: 20675638.0000 - mae: 1708.3080 - val_loss: 20259994.0000 - val_mae: 1519.7375
Epoch 20/50
5000/5000 - 13s - 3ms/step - loss: 20675022.0000 - mae: 1708.6954 - val_loss: 20156034.0000 - val_mae: 1613.8207
Epoch 21/50
5000/5000 - 13s - 3ms/step - loss: 20676422.0000 - mae: 1707.9432 - val_loss: 20467258.0000 - val_mae: 2089.2681
Epoch 22/50
5000/5000 - 13s - 3ms/step - loss: 20669056.0000 - mae: 1712.4094 - val_loss: 20128744.0000 - val_mae: 1621.6923
Epoch 23/50
5000/5000 - 13s - 3ms/step - loss: 20653052.0000 - mae: 1714.0190 - val_loss: 20153240.0000 - val_mae: 1568.2733
Epoch 24/50
5000/5000 - 13s - 3ms/step - loss: 20635172.0000 - mae: 1719.1346 - val_loss: 20029448.0000 - val_mae: 1695.4426
Epoch 25/50
5000/5000 - 12s - 2ms/step - loss: 20590564.0000 - mae: 1729.9852 - val_loss: 20144554.0000 - val_mae: 1523.4572
Epoch 26/50
5000/5000 - 13s - 3ms/step - loss: 20528016.0000 - mae: 1751.6534 - val_loss: 20005992.0000 - val_mae: 1605.4667
Epoch 27/50
5000/5000 - 13s - 3ms/step - loss: 20508928.0000 - mae: 1763.3318 - val_loss: 19867338.0000 - val_mae: 1949.3608
Epoch 28/50
5000/5000 - 13s - 3ms/step - loss: 20503820.0000 - mae: 1766.2362 - val_loss: 20044260.0000 - val_mae: 1572.9192
Epoch 29/50
5000/5000 - 13s - 3ms/step - loss: 20480218.0000 - mae: 1772.8602 - val_loss: 19827190.0000 - val_mae: 1802.0504
Epoch 30/50
5000/5000 - 12s - 2ms/step - loss: 20471234.0000 - mae: 1777.8156 - val_loss: 19968990.0000 - val_mae: 1614.2615
Epoch 31/50
5000/5000 - 13s - 3ms/step - loss: 20484786.0000 - mae: 1774.4596 - val_loss: 19928996.0000 - val_mae: 1644.4565
Epoch 32/50
5000/5000 - 13s - 3ms/step - loss: 20476312.0000 - mae: 1777.1385 - val_loss: 20052408.0000 - val_mae: 1566.8444
Epoch 33/50
5000/5000 - 13s - 3ms/step - loss: 20463176.0000 - mae: 1779.8022 - val_loss: 19892392.0000 - val_mae: 1673.5630
Epoch 34/50
5000/5000 - 12s - 2ms/step - loss: 20470524.0000 - mae: 1777.5062 - val_loss: 19870976.0000 - val_mae: 1718.2756
Epoch 35/50
5000/5000 - 22s - 4ms/step - loss: 20467820.0000 - mae: 1778.6694 - val_loss: 19928970.0000 - val_mae: 1656.7886
Epoch 36/50
5000/5000 - 13s - 3ms/step - loss: 20458998.0000 - mae: 1781.4780 - val_loss: 19851224.0000 - val_mae: 1728.8716
Epoch 37/50
5000/5000 - 13s - 3ms/step - loss: 20463482.0000 - mae: 1776.4680 - val_loss: 20117446.0000 - val_mae: 1533.5239
Epoch 38/50
5000/5000 - 13s - 3ms/step - loss: 20464346.0000 - mae: 1780.0150 - val_loss: 20118398.0000 - val_mae: 1522.9460
Epoch 39/50
5000/5000 - 12s - 2ms/step - loss: 20461004.0000 - mae: 1782.9250 - val_loss: 19821904.0000 - val_mae: 1911.3574
Epoch 40/50
5000/5000 - 12s - 2ms/step - loss: 20463628.0000 - mae: 1781.1216 - val_loss: 20152496.0000 - val_mae: 1505.1176
Epoch 41/50
5000/5000 - 12s - 2ms/step - loss: 20455844.0000 - mae: 1781.3204 - val_loss: 20022172.0000 - val_mae: 2116.2783
Epoch 42/50
5000/5000 - 12s - 2ms/step - loss: 20454176.0000 - mae: 1781.7058 - val_loss: 19816754.0000 - val_mae: 1822.4220
Epoch 43/50
5000/5000 - 12s - 2ms/step - loss: 20456246.0000 - mae: 1782.9756 - val_loss: 19944182.0000 - val_mae: 1630.7279
Epoch 44/50
5000/5000 - 12s - 2ms/step - loss: 20459198.0000 - mae: 1781.0610 - val_loss: 19834110.0000 - val_mae: 1754.5938
Epoch 45/50
5000/5000 - 12s - 2ms/step - loss: 20450534.0000 - mae: 1784.9576 - val_loss: 19825242.0000 - val_mae: 1915.0854
Epoch 46/50
5000/5000 - 15s - 3ms/step - loss: 20447576.0000 - mae: 1784.9749 - val_loss: 19804928.0000 - val_mae: 1779.0690
Epoch 47/50
5000/5000 - 21s - 4ms/step - loss: 20449376.0000 - mae: 1786.1440 - val_loss: 19841128.0000 - val_mae: 1728.2670
Epoch 48/50
5000/5000 - 32s - 6ms/step - loss: 20442094.0000 - mae: 1786.2704 - val_loss: 19930596.0000 - val_mae: 1626.6989
Epoch 49/50
5000/5000 - 12s - 2ms/step - loss: 20443820.0000 - mae: 1786.9036 - val_loss: 19821944.0000 - val_mae: 1884.9220
Epoch 50/50
5000/5000 - 12s - 2ms/step - loss: 20454768.0000 - mae: 1786.2712 - val_loss: 19930728.0000 - val_mae: 1638.6222
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step
Model 2 Evaluation -> MAE: 1670.98, R2: -0.01
Total Feature Weight: 2859.6199
Feature Importance for Model 2:
    Feature  Absolute_Importance  Relative_Importance
0       tag          1905.281006            66.627075
2      year           520.197998            18.191158
3  compound           419.968414            14.686163
1    artist            14.172516             0.495608
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step
Exporting structured data to BigQuery...
Tabla user_input_predictions_genre0 exportada a Big Query con éxito
Tabla views_pie_genre_year0 exportada a Big Query con éxito
Tabla views_total_year0 exportada a Big Query con éxito
Tabla sent_avg_genre_year0 exportada a Big Query con éxito
Tabla sent_avg_year0 exportada a Big Query con éxito
Tabla sent_radar_all_time0 exportada a Big Query con éxito
Tabla word_treemap0 exportada a Big Query con éxito
Tabla feature_importance_model0 exportada a Big Query con éxito
Todas las tablas estructuradas y exportadas a Big Query con éxito"""