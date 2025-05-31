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
user_input_df = load_csv_from_gcs(GCS_BUCKET, "4_song_user_input_genre_csv.csv")  # Entrada de datos del usuario

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
print(f"🔹 Total Feature Weight: {total_weight:.4f}")
print("🔹 Feature Importance for Model 2:")
print(feature_importance_df)

# Predecir visitas para canciones del usuario
user_input_views = user_input_df[['tag', 'artist', 'year', 'predicted_compound']]
user_input_df['predicted_views'] = model2_nn.predict(user_input_views)

# === EXPORTAR A BIG QUERY ===
print("Exporting structured data to BigQuery...")

load_dataframe_to_bigquery(user_input_df[['title', 'predicted_compound', 'predicted_views']], "user_input_predictions_genre")
load_dataframe_to_bigquery(views_pie_genre_year[['year', 'tag', 'views']], "views_pie_genre_year")
load_dataframe_to_bigquery(views_total_year[['year', 'total_views']], "views_total_year")
load_dataframe_to_bigquery(sent_avg_genre_year, "sent_avg_genre_year")
load_dataframe_to_bigquery(sent_avg_year[['year', 'avg_compound_sentiment']], "sent_avg_year")
load_dataframe_to_bigquery(sent_radar_all_time, "sent_radar_all_time")
load_dataframe_to_bigquery(word_treemap[['year', 'tag', 'word', 'frequency']], "word_treemap")
load_dataframe_to_bigquery(feature_importance_df, "feature_importance_model")

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
2025-03-08 16:51:20.761239: I external/local_xla/xla/tsl/cuda/cudart_stub.cc:32] Could not find cuda drivers on your machine, GPU will not be used.
2025-03-08 16:51:20.785294: I external/local_xla/xla/tsl/cuda/cudart_stub.cc:32] Could not find cuda drivers on your machine, GPU will not be used.
2025-03-08 16:51:20.824668: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1741452680.895854   30604 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1741452680.920024   30604 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2025-03-08 16:51:21.028242: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
Cargando datasets desde Google Cloud Storage...
Aplicando análisis de sentimiento a analytics dataset...
Aplicando análisis de sentimiento a predictions dataset...
/var/tmp/ipykernel_30604/2439335250.py:148: DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
  .apply(get_top_words)
Encoding categorical data for model training...
Training sentiment prediction model...
2025-03-08 17:08:20.986508: E external/local_xla/xla/stream_executor/cuda/cuda_driver.cc:152] failed call to cuInit: INTERNAL: CUDA error: Failed call to cuInit: UNKNOWN ERROR (303)
Epoch 1/20
10000/10000 - 68s - 7ms/step - loss: 0.3131 - mae: 0.3838 - val_loss: 0.2685 - val_mae: 0.3393
Epoch 2/20
10000/10000 - 66s - 7ms/step - loss: 0.2534 - mae: 0.3211 - val_loss: 0.2624 - val_mae: 0.3298
Epoch 3/20
10000/10000 - 66s - 7ms/step - loss: 0.2164 - mae: 0.2846 - val_loss: 0.2642 - val_mae: 0.3067
Epoch 4/20
10000/10000 - 66s - 7ms/step - loss: 0.1789 - mae: 0.2469 - val_loss: 0.2710 - val_mae: 0.3176
Epoch 5/20
10000/10000 - 67s - 7ms/step - loss: 0.1506 - mae: 0.2202 - val_loss: 0.2817 - val_mae: 0.3133
Epoch 6/20
10000/10000 - 67s - 7ms/step - loss: 0.1323 - mae: 0.2028 - val_loss: 0.2862 - val_mae: 0.3094
Epoch 7/20
10000/10000 - 67s - 7ms/step - loss: 0.1169 - mae: 0.1878 - val_loss: 0.2889 - val_mae: 0.3102
Epoch 8/20
10000/10000 - 68s - 7ms/step - loss: 0.1071 - mae: 0.1786 - val_loss: 0.2950 - val_mae: 0.3121
Epoch 9/20
10000/10000 - 68s - 7ms/step - loss: 0.0988 - mae: 0.1700 - val_loss: 0.2956 - val_mae: 0.3199
Epoch 10/20
10000/10000 - 70s - 7ms/step - loss: 0.0916 - mae: 0.1627 - val_loss: 0.2999 - val_mae: 0.3109
Epoch 11/20
10000/10000 - 68s - 7ms/step - loss: 0.0856 - mae: 0.1567 - val_loss: 0.3020 - val_mae: 0.3123
Epoch 12/20
10000/10000 - 69s - 7ms/step - loss: 0.0793 - mae: 0.1505 - val_loss: 0.3086 - val_mae: 0.3116
Epoch 13/20
10000/10000 - 65s - 7ms/step - loss: 0.0761 - mae: 0.1474 - val_loss: 0.3105 - val_mae: 0.3203
Epoch 14/20
10000/10000 - 65s - 7ms/step - loss: 0.0724 - mae: 0.1433 - val_loss: 0.3088 - val_mae: 0.3130
Epoch 15/20
10000/10000 - 65s - 7ms/step - loss: 0.0696 - mae: 0.1398 - val_loss: 0.3095 - val_mae: 0.3162
Epoch 16/20
10000/10000 - 64s - 6ms/step - loss: 0.0668 - mae: 0.1366 - val_loss: 0.3064 - val_mae: 0.3167
Epoch 17/20
10000/10000 - 64s - 6ms/step - loss: 0.0644 - mae: 0.1339 - val_loss: 0.3075 - val_mae: 0.3127
Epoch 18/20
10000/10000 - 65s - 6ms/step - loss: 0.0624 - mae: 0.1317 - val_loss: 0.3088 - val_mae: 0.3160
Epoch 19/20
10000/10000 - 67s - 7ms/step - loss: 0.0608 - mae: 0.1296 - val_loss: 0.3108 - val_mae: 0.3165
Epoch 20/20
10000/10000 - 65s - 7ms/step - loss: 0.0580 - mae: 0.1265 - val_loss: 0.3121 - val_mae: 0.3111
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 8s 5ms/step
Model 1 Evaluation -> MAE: 0.3110, R2: 0.5885
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 138ms/step
Training views prediction model...
Epoch 1/50
5000/5000 - 23s - 5ms/step - loss: 20787050.0000 - mae: 1703.2394 - val_loss: 20281376.0000 - val_mae: 1897.9965
Epoch 2/50
5000/5000 - 18s - 4ms/step - loss: 20726848.0000 - mae: 1699.4749 - val_loss: 20227488.0000 - val_mae: 1584.9670
Epoch 3/50
5000/5000 - 13s - 3ms/step - loss: 20726622.0000 - mae: 1697.0554 - val_loss: 20183558.0000 - val_mae: 1759.6008
Epoch 4/50
5000/5000 - 12s - 2ms/step - loss: 20724796.0000 - mae: 1698.4412 - val_loss: 20207824.0000 - val_mae: 1606.1545
Epoch 5/50
5000/5000 - 12s - 2ms/step - loss: 20721234.0000 - mae: 1698.4412 - val_loss: 20175718.0000 - val_mae: 1675.4216
Epoch 6/50
5000/5000 - 12s - 2ms/step - loss: 20718506.0000 - mae: 1698.1486 - val_loss: 20170428.0000 - val_mae: 1714.5212
Epoch 7/50
5000/5000 - 13s - 3ms/step - loss: 20716514.0000 - mae: 1699.2190 - val_loss: 20174734.0000 - val_mae: 1663.9592
Epoch 8/50
5000/5000 - 13s - 3ms/step - loss: 20714386.0000 - mae: 1699.8882 - val_loss: 20176764.0000 - val_mae: 1650.7990
Epoch 9/50
5000/5000 - 12s - 2ms/step - loss: 20707360.0000 - mae: 1701.4762 - val_loss: 20188644.0000 - val_mae: 1621.3650
Epoch 10/50
5000/5000 - 12s - 2ms/step - loss: 20708456.0000 - mae: 1700.8494 - val_loss: 20182964.0000 - val_mae: 1627.7325
Epoch 11/50
5000/5000 - 12s - 2ms/step - loss: 20708546.0000 - mae: 1701.5367 - val_loss: 20250364.0000 - val_mae: 1544.9733
Epoch 12/50
5000/5000 - 12s - 2ms/step - loss: 20707412.0000 - mae: 1700.9938 - val_loss: 20176890.0000 - val_mae: 1631.1139
Epoch 13/50
5000/5000 - 19s - 4ms/step - loss: 20701976.0000 - mae: 1702.2791 - val_loss: 20192216.0000 - val_mae: 1601.7515
Epoch 14/50
5000/5000 - 11s - 2ms/step - loss: 20693036.0000 - mae: 1704.7596 - val_loss: 20163318.0000 - val_mae: 1645.1340
Epoch 15/50
5000/5000 - 11s - 2ms/step - loss: 20697296.0000 - mae: 1704.1686 - val_loss: 20222464.0000 - val_mae: 1561.6355
Epoch 16/50
5000/5000 - 10s - 2ms/step - loss: 20687156.0000 - mae: 1704.9216 - val_loss: 20141224.0000 - val_mae: 1717.8696
Epoch 17/50
5000/5000 - 11s - 2ms/step - loss: 20686390.0000 - mae: 1705.8302 - val_loss: 20161312.0000 - val_mae: 1629.7892
Epoch 18/50
5000/5000 - 11s - 2ms/step - loss: 20679384.0000 - mae: 1707.3422 - val_loss: 20128642.0000 - val_mae: 1710.0212
Epoch 19/50
5000/5000 - 12s - 2ms/step - loss: 20676918.0000 - mae: 1709.0616 - val_loss: 20128586.0000 - val_mae: 1666.7375
Epoch 20/50
5000/5000 - 11s - 2ms/step - loss: 20666674.0000 - mae: 1710.5861 - val_loss: 20144378.0000 - val_mae: 1611.2791
Epoch 21/50
5000/5000 - 11s - 2ms/step - loss: 20652350.0000 - mae: 1713.4941 - val_loss: 20084942.0000 - val_mae: 1736.4091
Epoch 22/50
5000/5000 - 13s - 3ms/step - loss: 20639036.0000 - mae: 1718.6538 - val_loss: 20194248.0000 - val_mae: 1527.8630
Epoch 23/50
5000/5000 - 11s - 2ms/step - loss: 20614448.0000 - mae: 1723.2520 - val_loss: 20016336.0000 - val_mae: 1674.2845
Epoch 24/50
5000/5000 - 11s - 2ms/step - loss: 20572976.0000 - mae: 1737.5784 - val_loss: 20025046.0000 - val_mae: 1930.5400
Epoch 25/50
5000/5000 - 22s - 4ms/step - loss: 20530732.0000 - mae: 1751.7692 - val_loss: 20069126.0000 - val_mae: 1553.9497
Epoch 26/50
5000/5000 - 12s - 2ms/step - loss: 20499524.0000 - mae: 1764.9750 - val_loss: 19916298.0000 - val_mae: 1886.7004
Epoch 27/50
5000/5000 - 12s - 2ms/step - loss: 20490076.0000 - mae: 1768.0356 - val_loss: 20100486.0000 - val_mae: 2124.4172
Epoch 28/50
5000/5000 - 12s - 2ms/step - loss: 20483068.0000 - mae: 1772.4351 - val_loss: 19847066.0000 - val_mae: 1724.9417
Epoch 29/50
5000/5000 - 12s - 2ms/step - loss: 20484238.0000 - mae: 1770.9274 - val_loss: 19981710.0000 - val_mae: 1609.2059
Epoch 30/50
5000/5000 - 12s - 2ms/step - loss: 20473820.0000 - mae: 1775.4468 - val_loss: 19868890.0000 - val_mae: 1671.9666
Epoch 31/50
5000/5000 - 12s - 2ms/step - loss: 20479160.0000 - mae: 1775.2454 - val_loss: 20096856.0000 - val_mae: 1533.4456
Epoch 32/50
5000/5000 - 12s - 2ms/step - loss: 20467336.0000 - mae: 1775.8350 - val_loss: 20185946.0000 - val_mae: 2217.5215
Epoch 33/50
5000/5000 - 12s - 2ms/step - loss: 20460692.0000 - mae: 1781.0046 - val_loss: 20415456.0000 - val_mae: 2291.9558
Epoch 34/50
5000/5000 - 12s - 2ms/step - loss: 20459866.0000 - mae: 1778.6595 - val_loss: 19810980.0000 - val_mae: 1773.0883
Epoch 35/50
5000/5000 - 12s - 2ms/step - loss: 20458942.0000 - mae: 1779.5920 - val_loss: 20176856.0000 - val_mae: 2211.7363
Epoch 36/50
5000/5000 - 12s - 2ms/step - loss: 20466438.0000 - mae: 1780.5476 - val_loss: 20133758.0000 - val_mae: 1512.4286
Epoch 37/50
5000/5000 - 13s - 3ms/step - loss: 20458498.0000 - mae: 1780.0828 - val_loss: 19890502.0000 - val_mae: 1665.8278
Epoch 38/50
5000/5000 - 13s - 3ms/step - loss: 20461560.0000 - mae: 1781.2552 - val_loss: 19874062.0000 - val_mae: 1699.9062
Epoch 39/50
5000/5000 - 12s - 2ms/step - loss: 20466004.0000 - mae: 1780.4298 - val_loss: 19993282.0000 - val_mae: 1584.7797
Epoch 40/50
5000/5000 - 13s - 3ms/step - loss: 20458410.0000 - mae: 1783.1012 - val_loss: 20037800.0000 - val_mae: 1564.8792
Epoch 41/50
5000/5000 - 13s - 3ms/step - loss: 20456134.0000 - mae: 1782.1288 - val_loss: 20019856.0000 - val_mae: 1570.2997
Epoch 42/50
5000/5000 - 13s - 3ms/step - loss: 20462316.0000 - mae: 1781.6490 - val_loss: 19896328.0000 - val_mae: 1667.0756
Epoch 43/50
5000/5000 - 13s - 3ms/step - loss: 20452870.0000 - mae: 1785.0044 - val_loss: 19899592.0000 - val_mae: 1675.7753
Epoch 44/50
5000/5000 - 12s - 2ms/step - loss: 20450894.0000 - mae: 1783.3478 - val_loss: 19917404.0000 - val_mae: 1670.5358
Epoch 45/50
5000/5000 - 13s - 3ms/step - loss: 20447438.0000 - mae: 1786.4526 - val_loss: 19862378.0000 - val_mae: 1685.6620
Epoch 46/50
5000/5000 - 13s - 3ms/step - loss: 20447942.0000 - mae: 1785.6532 - val_loss: 19969538.0000 - val_mae: 1607.1321
Epoch 47/50
5000/5000 - 14s - 3ms/step - loss: 20445570.0000 - mae: 1786.6501 - val_loss: 19805778.0000 - val_mae: 1846.1162
Epoch 48/50
5000/5000 - 13s - 3ms/step - loss: 20450078.0000 - mae: 1783.4972 - val_loss: 19938256.0000 - val_mae: 1634.7343
Epoch 49/50
5000/5000 - 12s - 2ms/step - loss: 20447154.0000 - mae: 1785.6396 - val_loss: 19983492.0000 - val_mae: 1592.2974
Epoch 50/50
5000/5000 - 12s - 2ms/step - loss: 20441664.0000 - mae: 1788.5844 - val_loss: 19994586.0000 - val_mae: 1579.2194
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step
Model 2 Evaluation -> MAE: 1610.28, R2: -0.01
🔹 Total Feature Weight: 4093.5752
🔹 Feature Importance for Model 2:
    Feature  Absolute_Importance  Relative_Importance
0       tag          2678.000000            65.419586
2      year           799.619263            19.533518
3  compound           606.292358            14.810827
1    artist             9.663584             0.236067
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 31ms/step
Exporting structured data to BigQuery...
Tabla user_input_predictions_genre exportada a Big Query con éxito
Tabla views_pie_genre_year exportada a Big Query con éxito
Tabla views_total_year exportada a Big Query con éxito
Tabla sent_avg_genre_year exportada a Big Query con éxito
Tabla sent_avg_year exportada a Big Query con éxito
Tabla sent_radar_all_time exportada a Big Query con éxito
Tabla word_treemap exportada a Big Query con éxito
Tabla feature_importance_model exportada a Big Query con éxito
Todas las tablas estructuradas y exportadas a Big Query con éxito"""