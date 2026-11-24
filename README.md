# Segmentación de Videojuegos Reales en Steam mediante K-Means y Dashboard Interactivo

Proyecto integrador de minería de datos correspondiente a la **Unidad 3**, enfocado en el descubrimiento de segmentos reales en el catálogo de **Steam** utilizando el dataset público de Kaggle (`nikdavis/steam-store-games`), el algoritmo no supervisado **K-Means** en Python y un **Dashboard Interactivo en Streamlit**.

---

## Descripción del Proyecto

A partir de datos reales de **27,075 videojuegos** publicados en Steam (`datos/steam.csv`), este proyecto aplica el ciclo completo de análisis de datos:

1. **Exploración inicial (EDA):** inspección de `27,075` registros y `18` variables originales, cálculo de estadísticas descriptivas, detección de `15` valores nulos reales (`developer` y `publisher`), software no gamer con precios altos (hasta `421.99 GBP`), juegos sin tiempo registrado (`average_playtime == 0`) y valores extremos de horas promedio.
2. **Limpieza y preparación de datos:** imputación de nulos con `"Desconocido"`, verificación de duplicados exactos (`0`), conversión de fechas (`release_date` -> `anio_lanzamiento`), creación de variables derivadas (`calificacion_pct`, `horas_promedio`, `genero_principal`, `propietarios_est`), filtrado de software/valores extremos (`price <= 60`, `horas_promedio <= 60`, `total_resenas >= 20`, `average_playtime > 0`), codificación (`LabelEncoder`) y estandarización (`StandardScaler`), obteniendo **5,864 videojuegos reales limpios**.
3. **Modelado con K-Means:** evaluación de `K = 2` hasta `K = 7` mediante el Método del Codo y el Coeficiente de Silueta, selección de **`K = 4`** (`Inercia = 6,307.99`, `Silhouette = 0.3900`, `9` iteraciones), análisis de centroides y comparación frente a `K = 3`.
4. **Dashboard interactivo:** aplicación web en Streamlit con 2 páginas, 4 KPIs por página, 3 filtros laterales en tiempo real, gráfica de dispersión con centroides, comparativas, tabla descriptiva y conclusiones.

---

## Herramientas y Bibliotecas Utilizadas

* **Python 3** — Lenguaje principal de programación
* **Pandas** — Carga, limpieza, transformación y análisis tabular
* **Scikit-learn** — Preprocesamiento (`LabelEncoder`, `StandardScaler`), `KMeans` y `silhouette_score`
* **Matplotlib & Seaborn** — Visualización estadística exploratoria y representación de clústeres
* **Streamlit** — Desarrollo del dashboard web interactivo

---

## Estructura del Repositorio

```text
Proyecto_Unidad3/
├── datos/
│   ├── steam.csv                       # Dataset real original de Steam (27,075 registros, Kaggle)
│   ├── steam_procesado.csv             # Dataset limpio con clúster y segmento asignados (5,864 registros)
│   ├── centroides_kmeans.csv           # Centroides del modelo final K=4 en escala original
│   ├── metricas_evaluacion_k.csv       # Inercia y Silhouette Score para K=2..7
│   └── resumen_modelo.json             # Resumen estructurado de métricas del modelo
├── src/
│   └── analisis_kmeans.py              # Script lineal estilo práctica (EDA, limpieza, K-Means y gráficas)
├── dashboard/
│   └── app.py                          # Dashboard interactivo en Streamlit (2 páginas)
├── documentacion/                      # Carpeta destinada al documento PDF final del proyecto
├── requirements.txt                    # Dependencias del proyecto
├── .gitignore
└── README.md
```

---

## Instrucciones de Ejecución

### 1. Instalar dependencias

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Ejecutar el análisis y modelo K-Means

Este comando explora y limpia `datos/steam.csv`, muestra las gráficas en pantalla con `plt.show()`, entrena el modelo K-Means (`K = 4`) y guarda los resultados procesados en `datos/`:

```bash
python3 src/analisis_kmeans.py
```

### 3. Visualizar el Dashboard Interactivo

```bash
streamlit run dashboard/app.py
```

Una vez iniciado, abre el navegador en `http://localhost:8501` para explorar la **Página 1 (Análisis general)** y la **Página 2 (Segmentación mediante K-Means)**.

---

## Resumen de Segmentos Reales Identificados (K = 4)

| Clúster | Segmento | Juegos (%) | Precio Prom. (GBP) | Calificación Positiva | Horas Prom. |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **0** | **Económicos de Baja Aceptación** | 1,607 (27.40%) | 3.33 GBP | 50.90% | 3.15 h |
| **1** | **Económicos Bien Valorados** | 3,091 (52.71%) | 4.97 GBP | 83.41% | 3.77 h |
| **2** | **Gama Alta / Premium** | 839 (14.31%) | 20.58 GBP | 81.25% | 6.84 h |
| **3** | **Éxitos de Alta Retención** | 327 (5.58%) | 16.20 GBP | 79.46% | 33.52 h |

---

## Fuente del Dataset y Documentación

* **Fuente pública original:** Kaggle — *Steam Store Games (Clean dataset)* por Nik Davis (`nikdavis/steam-store-games`, archivo `steam.csv`).
* **Documentación final:** el PDF con el desarrollo completo del proyecto se alojará en la carpeta `documentacion/`.
