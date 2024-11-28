import streamlit as st
import joblib
import pandas as pd
import numpy as np
from surprise import SVD

def generate_recommendations_streamlit(user_id, svd, recommendations_data, games_data_merged, top_n=10):
    """
    Genera recomendaciones de juegos para un usuario específico utilizando un modelo SVD,
    adaptado para Streamlit.

    Args:
        user_id (int): El ID del usuario para el que se generarán las recomendaciones.
        svd (SVD): Modelo entrenado de Singular Value Decomposition para realizar predicciones.
        recommendations_data (DataFrame): DataFrame con usuarios, juegos y calificaciones.
        games_data_merged (DataFrame): DataFrame con detalles de los juegos, incluidas las columnas 'app_id' y 'title'.
        top_n (int): Número de recomendaciones a generar.

    Returns:
        list: Lista de diccionarios que contienen los juegos recomendados con sus títulos y calificaciones estimadas.
    """
    all_games = recommendations_data['app_id'].unique()
    user_games = recommendations_data[recommendations_data['user_id'] == user_id]['app_id'].unique()
    unrated_games = [game for game in all_games if game not in user_games]

    recommendations = [
        (game, svd.predict(user_id, game).est) for game in unrated_games
    ]
    recommendations.sort(key=lambda x: x[1], reverse=True)

    top_recommendations = recommendations[:top_n]

    # Crear lista de resultados con títulos
    result = []
    for rec in top_recommendations:
        filtered_titles = games_data_merged.loc[games_data_merged['app_id'] == rec[0], 'title']
        title = filtered_titles.iloc[0] if not filtered_titles.empty else "Unknown Title"
        result.append({"title": title, "score": rec[1]})

    return result

svd_sample = joblib.load("svd_sample.joblib")

# Configurar lista de ítems y usuarios
recommendations_data = pd.read_parquet('recommendations.gzip')
games_data_merged = pd.read_csv('games_data_merged.csv')

user_ids = recommendations_data['user_id'].unique()
app_ids = recommendations_data['app_id'].unique()

# Título
st.title("Sistema de Recomendación de Juegos de Steam")

# Selección de usuario
user_id = st.selectbox("Selecciona un usuario", user_ids)

# Número de recomendaciones
top_n = st.slider("Número de recomendaciones", min_value=1, max_value=20, value=10)

# Botón para generar recomendaciones
if st.button("Generar Recomendaciones"):
    recommendations = generate_recommendations_streamlit(user_id,svd_sample, games_data_merged, top_n)
    st.write("Tus recomendaciones:")
    for app_id, score in recommendations:
        st.write(f"Juego: {app_id}, Puntuación estimada: {score:.2f}")
