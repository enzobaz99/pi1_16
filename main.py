"""
este codigo es para una app que trabaja con datos de juegos de steam
"""

from fastapi import FastAPI
import pandas as pd
from sklearn.metrics.pairwise import  linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer



app=FastAPI(debug=True)

df = pd.read_csv('df_completo.csv')

@app.get('/Cantidad de items y porcentaje de contenido gratuito/')
def Developer(desarrollador: str) -> dict:
    # Filtrar el DataFrame por la empresa desarrolladora
    df_desarrolladora = df[df['developer'] == desarrollador]

    # Crear un DataFrame agrupado por año
    df_grouped = df_desarrolladora.groupby('release_date').agg({'id': 'count', 'price': 'mean'})

    # Renombrar las columnas
    df_grouped = df_grouped.rename(columns={'id': 'Cantidad de Items', 'price': 'Precio Promedio'})

    # Identificar las variantes de contenido gratuito
    variantes_contenido_gratuito = ['Free to Play', 'Free To Play', 'Play for Free!']

    # Calcular el porcentaje de contenido gratuito
    df_grouped['Contenido Free'] = (df_grouped['Precio Promedio'].apply(lambda x: any(variante in x for variante in variantes_contenido_gratuito))).sum() / df_grouped.shape[0] * 100

    # Resetear el índice y rellenar los valores NaN
    df_grouped = df_grouped.reset_index()
    df_grouped = df_grouped.fillna(0)

    # Formatear los valores como porcentaje
    df_grouped['Contenido Free'] = df_grouped['Contenido Free'].apply(lambda x: f'{x:.2f}%')

    return df_grouped

@app.get('/Cantidad de dinero gastado por jugador/')
def userdata(User_id: str) -> dict:
    # Filtrar el DataFrame por el User_id
    df_usuario = df[df['user_id'] == User_id]

    # Calcular la cantidad de dinero gastado por el usuario
    dinero_gastado = df_usuario['price'].sum()

    # Calcular el porcentaje de recomendación en base a reviews.recommend
    porcentaje_recomendacion = (df_usuario['recommend'].sum() / df_usuario.shape[0]) * 100

    # Calcular la cantidad de items
    cantidad_items = df_usuario['id'].nunique()

    # Crear un diccionario con los resultados
    resultado = {
        "User_id": User_id,
        "Dinero gastado": f"{dinero_gastado} USD",
        "% de recomendación": f"{porcentaje_recomendacion:.2f}%",
        "Cantidad de items": cantidad_items
    }

    return resultado

@app.get('/Cantidad de dinero gastado por jugador/')
def UserForGenre(genero: str) -> dict:
    # Filtrar el DataFrame por el género dado
    df_genero = df[df['genres'].str.contains(genero)]

    if df_genero.empty:
        return "No se encontraron datos para el género especificado."

    # Encontrar al usuario con más horas jugadas para el género
    usuario_max_horas = df_genero.loc[df_genero['playtime_forever'].idxmax()]['user_id']

    # Crear un DataFrame agrupado por año y sumar las horas jugadas
    df_grouped = df_genero.groupby('posted year')['playtime_forever'].sum().reset_index()

    # Crear una lista de acumulación de horas jugadas por año
    acumulacion_horas = [{"Año": year, "Horas": horas} for year, horas in zip(df_grouped['posted year'], df_grouped['playtime_forever'])]

    # Crear el diccionario de retorno
    resultado = {
        f"Usuario con más horas jugadas para Género {genero}": usuario_max_horas,
        "Horas jugadas": acumulacion_horas
    }

    return resultado

@app.get('/Cantidad de dinero gastado por jugador/')
def best_developer_year(año: int) -> dict:
    # Filtrar el DataFrame por el año dado
    df_año = df[df['posted year'] == año]

    if df_año.empty:
        return "No se encontraron datos para el año especificado."

    # Filtrar los juegos con recomendaciones positivas
    df_recomendados = df_año[df_año['recommend'] == True]

    # Agrupar por desarrollador y contar el número de juegos recomendados
    desarrolladores_recomendados = df_recomendados.groupby('developer')['id'].count().reset_index()

    # Ordenar por número de juegos recomendados de mayor a menor
    desarrolladores_recomendados = desarrolladores_recomendados.sort_values(by='id', ascending=False)

    # Tomar los 3 primeros desarrolladores
    top_3_desarrolladores = desarrolladores_recomendados.head(3)

    # Crear la lista de retorno en el formato especificado
    resultado = [{"Puesto 1": top_3_desarrolladores.iloc[0]['developer']},
                 {"Puesto 2": top_3_desarrolladores.iloc[1]['developer']},
                 {"Puesto 3": top_3_desarrolladores.iloc[2]['developer']}]

    return resultado

@app.get('/Cantidad de dinero gastado por jugador/')
def developer(desarrolladora: str) -> dict:
    # Filtrar el DataFrame por la desarrolladora
    df_desarrolladora = df[df['developer'] == desarrolladora]

    if df_desarrolladora.empty:
        return f"No se encontraron datos para la desarrolladora {desarrolladora}."

    # Contar la cantidad total de registros de reseñas con análisis positivo y negativo
    positivas = (df_desarrolladora['sentiment_score'] == 0).sum()
    negativas = (df_desarrolladora['sentiment_score'] == 2).sum()

    # Crear el diccionario de retorno en el formato especificado
    resultado = {desarrolladora: {'Positive': positivas, 'Negative': negativas}}

    return resultado

@app.get('/Cantidad de dinero gastado por jugador/')
def recomendacion_juego(id_producto: object) -> dict:
    # Filtrar el DataFrame para obtener los datos del juego con el ID especificado
    juego = df[df['id'] == id_producto]

    if juego.empty:
        return "No se encontraron datos para el juego con el ID especificado."

    # Crear un vectorizador TF-IDF para procesar los datos de texto (por ejemplo, el título y las etiquetas)
    tfidf_vectorizer = TfidfVectorizer()

    # Combinar los datos de texto relevantes en una sola columna
    df['texto_combinado'] = df['app_name'] + ' ' + df['tags']

    # Aplicar el vectorizador TF-IDF a los datos de texto combinados
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['texto_combinado'])

    # Calcular las similitudes de coseno entre los juegos
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Obtener el índice del juego ingresado
    idx = juego.index[0]

    # Obtener las similitudes del juego ingresado con otros juegos
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Ordenar los juegos por similitud en orden descendente
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Obtener los 5 juegos recomendados más similares
    top_juegos = sim_scores[1:6]

    # Obtener los IDs de los juegos recomendados
    juegos_recomendados = [df.iloc[juego[0]]['id'] for juego in top_juegos]

    return juegos_recomendados

@app.get('/Cantidad de dinero gastado por jugador/')
def recomendacion_usuario(id_usuario: object) -> dict:
    # Filtrar el DataFrame para obtener las reseñas del usuario con el ID especificado
    reseñas_usuario = df[df['user_id'] == id_usuario]

    if reseñas_usuario.empty:
        return "No se encontraron reseñas para el usuario con el ID especificado."

    # Obtener los juegos que el usuario ha revisado
    juegos_usuario = reseñas_usuario['item_id_y'].unique()

    # Filtrar el DataFrame de juegos para excluir los juegos revisados por el usuario
    df_juegos = df[~df['item_id_y'].isin(juegos_usuario)]

    # Crear un vectorizador TF-IDF para procesar los
    # datos de texto (por ejemplo, el título y las etiquetas)
    tfidf_vectorizer = TfidfVectorizer()

    # Combinar los datos de texto relevantes en una sola columna
    df['texto_combinado'] = df['app_name'] + ' ' + df['tags']

    # Aplicar el vectorizador TF-IDF a los datos de texto combinados
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['texto_combinado'])

    # Calcular las similitudes de coseno entre los juegos y las reseñas del usuario
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Crear un diccionario para almacenar las sumas de similitudes de juegos
    sumas_similitudes = {}

    # Iterar a través de las reseñas del usuario
    for juego in juegos_usuario:
        idx = df_juegos[df['item_id_y'] == juego].index[0]
        similitudes_juego = list(enumerate(cosine_sim[idx]))
        for juego_sim in similitudes_juego:
            juego_id = df.iloc[juego_sim[0]]['item_id_y']
            similitud = juego_sim[1]
            if juego_id not in sumas_similitudes:
                sumas_similitudes[juego_id] = similitud
            else:
                sumas_similitudes[juego_id] += similitud

    # Ordenar los juegos por suma de similitudes en orden descendente
    juegos_recomendados = sorted(sumas_similitudes.items(), key=lambda x: x[1], reverse=True)

    # Obtener los 5 juegos recomendados más similares
    juegos_recomendados = juegos_recomendados[:5]

    return [juego[0] for juego in juegos_recomendados]
