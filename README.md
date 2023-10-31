# <h1 align=center> **PROYECTO INDIVIDUAL Nº1** </h1>
![](https://trycore.co/cms/wp-content/uploads/2020/02/que-es-machine-learning-servicio-implementar-inteligencia-artificial-min.jpg)
Este proyecto consiste en crear una API que utiliza un modelo de recomendación para Steam, una plataforma multinacional de videojuegos, basado en Machine Learning. El objetivo es crear un sistema de recomendación de videojuegos para usuarios. La API ofrece una interfaz intuitiva para que los usuarios puedan obtener informacion para el sistema de recomendacion y datos sobre generos o fechas puntuales.

## **Herramientas Utilizadas**
+ Pandas
+ Matplotlib
+ Numpy
+ Seaborn
+ Wordcloud
+ NLTK
+ Uvicorn
+ Render
+ FastAPI
+ Python
+ Scikit-Learn
## **Pasos**
**`ETL`**: donde se nos pidio desanidar harchivos json.

**`desarrollar una API`**: en la api teniamos que desarrollar funciones:
+ def developer( desarrollador : str ): Cantidad de items y porcentaje de contenido Free por año según empresa desarrolladora.
+ def userdata( User_id : str ): Debe devolver cantidad de dinero gastado por el usuario, el porcentaje de recomendación en base a reviews.recommend y cantidad de items.
+ def UserForGenre( genero : str ): Debe devolver el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año de lanzamiento.
+ def best_developer_year( año : int ): Devuelve el top 3 de desarrolladores con juegos MÁS recomendados por usuarios para el año dado.
+ def developer_reviews_analysis( desarrolladora : str ): Según el desarrollador, se devuelve un diccionario con el nombre del desarrollador como llave y una lista con la cantidad total de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento como valor positivo o negativo.

**`EDA`**: donde mostraramos datos un poco mas estadisticos para mayor comprencion.

**`Modelos`**: aqui se nos pidio relizar modelos de aprendizaje donde teniamos que realizar funciones que serian estas:
+ def recomendacion_juego( id de producto ) Ingresando el id de producto, deberíamos recibir una lista con 5 juegos recomendados similares al ingresado.
+ def recomendacion_usuario( id de usuario ) Ingresando el id de un usuario, deberíamos recibir una lista con 5 juegos recomendados para dicho usuario.

## **Links:**
- [Deploy de la API en Render](http://127.0.0.1:8000/docs#/)
- [Video de youtube explicando el proyecto](https://youtu.be/ly8YC4zy17c?si=IYD5UIEinRrfHTJi)
