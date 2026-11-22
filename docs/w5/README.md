# Workshop 5 — GAN Report (LaTeX / Overleaf)

Este directorio contiene el artículo académico del Workshop 5 (GANs sobre MovieLens 32M).

## Archivos
- `group7_workshop5.tex` — fuente LaTeX del paper (dos columnas, estilo de los Workshops 2 y 4).
- `figures/` — aquí van las figuras PNG que genera el pipeline en Kaggle (ver abajo).

## Cómo compilar en Overleaf
1. Entra a https://www.overleaf.com y crea un proyecto en blanco (**New Project → Blank Project**).
2. Sube `group7_workshop5.tex` (**Upload**). Puedes borrar el `main.tex` que crea Overleaf por defecto, o renombrar este a `main.tex`.
3. Crea una carpeta `figures/` dentro del proyecto y sube ahí los 5 PNG (ver lista).
4. En el menú de compilación elige **pdfLaTeX** y pulsa **Recompile**. El PDF sale a la derecha.
5. Descarga el PDF y renómbralo a `group7_workshop5.pdf` para la entrega.

> El archivo usa `thebibliography` embebido, así que NO necesitas BibTeX ni archivos `.bib`.

## Figuras necesarias (colócalas en `docs/w5/figures/`)
Las genera el pipeline de W5 en `runs/workshop5/`. Descárgalas del Output de Kaggle y súbelas con EXACTAMENTE estos nombres:

| Archivo en figures/ | Lo produce |
|---|---|
| `training_curves.png` | `src/w5_figures.py` (curvas de pérdida G/D y W_dist) |
| `grid_real_vs_wgan_gp.png` | `src/generate_samples.py` |
| `pca_real_vs_generated.png` | `src/generate_samples.py` |
| `samples_grid_wgan.png` | `src/w5_figures.py` (muestras cada 10 épocas) |

Figuras opcionales (mencionadas en el texto, no obligatorias para compilar):
`grid_real_vs_vanilla.png`, `feature_dist_vanilla.png`, `feature_dist_wgan_gp.png`,
`samples_grid_vanilla.png`.

> Si falta alguna figura al compilar, LaTeX avisará con un recuadro, pero el resto del PDF se
> genera igual. Sube al menos las cuatro de la tabla para que el paper quede completo.

## Fuentes de los números
Todas las cifras del paper salen de ejecutar el código en Kaggle:
- `runs/workshop5/gan_quality_metrics.json` (Tabla de calidad)
- `runs/workshop5/augmentation_results.json` (Tabla de augmentation)
- `runs/workshop5/cumulative_comparison.csv` (Tabla acumulativa W1–W5)
- Cifras de W1/W2/W4: tomadas de los papers del equipo.
