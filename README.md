# Workshop 1 — Sistema de Recomendación con MovieLens 32M

Taller de Machine Learning: análisis exploratorio, preparación de datos,
ingeniería de características y evaluación de modelos base supervisados sobre
MovieLens 32M.

---

## Estructura del Repositorio

```text
Machine_Learning/
├── src/
│   ├── data_loader.py     # Carga, submuestreo, filtrado, IDs contiguos y split temporal
│   ├── features.py        # Ingeniería de características (estadísticos, sesgos, géneros, antigüedad)
│   └── baselines.py       # Entrenamiento y evaluación de modelos base (regresión y clasificación)
├── notebooks/
│   └── eda.ipynb          # Cuaderno completo de EDA con visualizaciones y análisis
├── data/
│   └── splits.json        # Índices y metadatos de partición train/val/test para reproducibilidad
├── dataSet/               # Datos crudos de MovieLens 32M (no versionados en Git)
├── checkpoints/           # Pesos de modelos guardados (talleres futuros)
├── runs/                  # Registros y gráficos de evaluación (residuales y matrices de confusión)
├── requirements.txt       # Dependencias fijadas del entorno virtual
└── README.md
```

---

## Dataset

**MovieLens 32M** (GroupLens Research, University of Minnesota).

- **Fuente oficial:** https://grouplens.org/datasets/movielens/32m/
- **Archivo:** `ml-32m.zip` (~250 MB comprimido, ~1 GB descomprimido)
- **Contenido:** 32.000.204 ratings, 200.948 usuarios, 87.585 películas, enero 1995 – octubre 2023.
- **Licencia:** uso académico (ver `README.txt` del dataset).

### Descarga y ubicación

1. Descarga `ml-32m.zip` desde la fuente oficial o el enlace alternativo.
2. Descomprímelo en la carpeta `dataSet/` con la siguiente estructura:

```text
Machine_Learning/
└── dataSet/
    ├── ratings.csv    # userId, movieId, rating, timestamp
    ├── movies.csv     # movieId, title, genres
    ├── tags.csv       # userId, movieId, tag, timestamp
    └── links.csv      # movieId, imdbId, tmdbId
```

---

## Configuración del Entorno

Crear y activar el entorno virtual e instalar las dependencias desde `requirements.txt`:

```bash
python3 -m venv .venv
source .venv/bin/activate      # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Comandos de Reproducción

1. **Generar particiones reproducibles (`data/splits.json`):**

```bash
.venv/bin/python -m src.data_loader
```

2. **Entrenar y evaluar los modelos base (regresión y clasificación):**

```bash
.venv/bin/python -m src.baselines
```

Esto imprime las métricas en consola y guarda los gráficos en `runs/`:
- `runs/residuals_regression.png` — análisis de residuales ($y - \hat{y}$) para los modelos de regresión.
- `runs/confusion_matrices.png` — matrices de confusión para los clasificadores sobre `like`.

3. **Análisis Exploratorio de Datos (EDA):**
   - Abrir y ejecutar `notebooks/eda.ipynb` seleccionando el kernel `.venv`.

---

## Métricas más importantes para la tarea

- **Regresión (`rating` en escala 0.5–5.0):**
  - **RMSE / MSE** y **MAE** son las métricas principales porque miden directamente el error en estrellas de la predicción de `rating` (RMSE penaliza más los errores grandes, mientras MAE refleja el error típico absoluto).
  - **$R^2$** complementa la lectura indicando qué proporción de la varianza de los ratings explica cada modelo respecto a predecir la media global.
- **Clasificación binaria (`like = 1` si `rating >= 4.0`):**
  - **AUC-ROC** es la métrica más importante porque evalúa la capacidad de ordenamiento/discriminación del modelo entre ítems que gustarán y los que no a través de todos los umbrales de probabilidad, lo cual es clave en recomendación.
  - **F1-score** (junto con **precisión** y **recall**) es fundamental para equilibrar falsos positivos (recomendar películas que no gustan) y falsos negativos (omitir películas relevantes), siendo más informativa que `accuracy` cuando las clases no están perfectamente balanceadas.
