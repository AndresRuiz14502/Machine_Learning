@'
# Workshop 1 — Sistema de Recomendación con MovieLens 32M

Taller de Machine Learning: construcción de un sistema de recomendación
basado en filtrado colaborativo, desde el análisis exploratorio hasta la
evaluación de modelos de factorización de matrices.


---

## Dataset

**MovieLens 32M** (GroupLens Research, University of Minnesota).

- **Fuente oficial:** https://grouplens.org/datasets/movielens/32m/
- **Archivo:** `ml-32m.zip` (~250 MB comprimido, ~1 GB descomprimido)
- **Contenido:** 32.000.204 ratings, 200.948 usuarios, 87.585 películas,
  enero 1995 – octubre 2023.
- **Licencia:** uso académico (ver README del dataset).

### Descarga y ubicación

1. Descargar `ml-32m.zip` desde el enlace oficial o del siguiente enlace: https://drive.google.com/drive/folders/1RB8vg3Sc2HD2XlrOCdaUIwm5cEbPXRQ4?usp=sharing 
## Instalación de los datos

1. Descarga el dataset MovieLens desde https://grouplens.org/datasets/movielens/

2. Descomprímelo en la carpeta `dataSet/` con la siguiente estructura:

```text
Machine_Learning/
└── dataSet/
    ├── ratings.csv    # userId, movieId, rating, timestamp
    ├── movies.csv     # movieId, title, genres
    ├── tags.csv       # userId, movieId, tag, timestamp
    └── links.csv      # movieId, imdbId, tmdbId
```

3. Verifica que los archivos estén en la ruta correcta antes de ejecutar el cuaderno.
