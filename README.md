# Workshop 1 — Sistema de Recomendación con MovieLens 32M

Taller de Machine Learning: construcción de un sistema de recomendación
basado en filtrado colaborativo, desde el análisis exploratorio hasta la
evaluación de modelos de factorización de matrices.

---

## Estructura del Repositorio

```text
Machine_Learning/
├── src/
│   ├── __init__.py
│   ├── data_loader.py     # Carga, submuestreo, filtrado, IDs contiguos y split temporal
│   ├── features.py        # Ingeniería de características (estadísticos, sesgos, géneros, antigüedad)
│   └── baselines.py       # Métricas y baselines (Popularity, Bias, kNN, Ridge, Lasso, Árboles)
├── notebooks/
│   ├── eda.ipynb          # Cuaderno completo de EDA con visualizaciones y análisis
│   └── WorkShop1.ipynb    # Flujo de modelado, FunkSVD, ajuste de hiperparámetros y evaluación
├── data/
│   └── splits.json        # Índices y metadatos de partición train/val/test para reproducibilidad
├── dataSet/               # Datos crudos de MovieLens 32M (no versionados en Git)
├── checkpoints/           # Pesos de modelos guardados (talleres futuros)
├── runs/                  # Registros de entrenamiento (wandb / TensorBoard)
├── requirements.txt       # Dependencias fijadas del entorno virtual
└── README.md
```

---

## Dataset

**MovieLens 32M** (GroupLens Research, University of Minnesota).

- **Fuente oficial:** https://grouplens.org/datasets/movielens/32m/
- **Enlace alternativo:** https://drive.google.com/drive/folders/1RB8vg3Sc2HD2XlrOCdaUIwm5cEbPXRQ4?usp=sharing
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

2. **Entrenar y evaluar los modelos base (`src/baselines.py`):**

```bash
.venv/bin/python -m src.baselines
```

3. **Explorar el EDA y ejecutar el modelado completo (`FunkSVD`):**
   - Abrir y ejecutar `notebooks/eda.ipynb` para el análisis exploratorio completo.
   - Abrir y ejecutar `notebooks/WorkShop1.ipynb` (seleccionando el kernel `.venv`) para correr el flujo completo con `FunkSVD` y la evaluación final.
