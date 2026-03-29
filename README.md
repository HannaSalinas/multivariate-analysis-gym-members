# 🏋️ Análisis Multivariado — Miembros de Gimnasio

**Autora:** Hanna Jineth Contreras Salinas  
**Programa:** Ingeniería de Software y Datos — IUDigital  
**Materia:** Estadística / Análisis de Datos  

---

## 📋 Descripción

Análisis exploratorio, descriptivo y multivariado completo sobre un dataset real de **970 miembros de gimnasio con 16 variables**. El proyecto cubre desde estadísticas descriptivas hasta reducción de dimensionalidad con PCA, pasando por análisis de correlaciones y visualizaciones profesionales.

---

## 🔬 ¿Qué hace este proyecto?

| Numeral | Técnica | Descripción |
|---------|---------|-------------|
| 1 | EDA completo | Estadísticas descriptivas, histogramas, boxplots, scatter plots y análisis de variables categóricas |
| 2 | Matriz de correlación | Heatmap de correlaciones entre todas las variables numéricas con interpretación |
| 3 | PCA | Reducción a 4 componentes principales, scree plot, heatmap de loadings y biplot |

---

## 🛠️ Tecnologías

| Librería | Uso |
|----------|-----|
| `pandas` | Carga, limpieza y manipulación del dataset Excel |
| `numpy` | Cálculos numéricos y estadísticos |
| `matplotlib` | Histogramas, scatter plots, scree plot, biplot |
| `seaborn` | Heatmaps de correlación y loadings, boxplots |
| `scikit-learn` | `StandardScaler` para normalización + `PCA` para reducción dimensional |

---

## 🚀 Cómo ejecutar

### 1. Instalar dependencias

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
```

### 2. Asegurarse de tener el dataset en el mismo directorio

```
📁 proyecto/
├── Analisis_multivariado_gimnasio.py
└── Miembros_gimnasio.xlsx
```

### 3. Ejecutar el script

```bash
python Analisis_multivariado_gimnasio.py
```

### 4. Gráficas generadas

El script produce 8 archivos PNG automáticamente:

```
1_histogramas_numericas.png      → Distribuciones de todas las variables numéricas
1_barras_categoricas.png         → Distribución de variables categóricas
1_boxplots_calorias.png          → Boxplots de calorías por tipo de entrenamiento
1_scatter_imc_grasa.png          → Relación IMC vs porcentaje de grasa
2_heatmap_correlacion.png        → Matriz de correlación completa
3_scree_plot_pca.png             → Varianza explicada por componente
3_heatmap_loadings.png           → Contribución de variables a cada componente
3_biplot_pca.png                 → Biplot PC1 vs PC2 coloreado por IMC
```

---

## 📊 Dataset

| Atributo | Valor |
|----------|-------|
| Registros | 970 (961 tras limpieza) |
| Variables | 16 (12 numéricas, 4 categóricas) |
| Fuente | Dataset académico de miembros de gimnasio |
| Formato | Excel (.xlsx) |

**Variables principales:** Edad, Peso, Altura, IMC, Duración_sesión, Calorías_quemadas, Frecuencia_entrenamiento, Porcentaje_grasa, Máx_BPM, Promedio_BPM, Reposo_BPM, Nivel_experiencia, Tipo_entrenamiento, Género, Categoría_IMC

---

## 📈 Resultados clave

### Análisis descriptivo
- Edad promedio: **38.7 años** (CV = 31.4% — dispersión moderada)
- IMC promedio: **24.9** — límite superior del rango normal
- Calorías quemadas: variable con **mayor variabilidad** (CV = 30.2%)
- Porcentaje de grasa: asimetría negativa → mayoría en rango medio-alto

### Correlaciones destacadas
- `Calorías_quemadas` ↔ `Duración_sesión`: correlación positiva fuerte
- `IMC` ↔ `Peso`: correlación muy alta esperada
- `Porcentaje_grasa` ↔ `Frecuencia_entrenamiento`: correlación negativa (más entrenamiento → menos grasa)

### PCA — 4 componentes
| Componente | Varianza | Interpreta |
|------------|----------|-----------|
| PC1 | ~25% | Volumen de entrenamiento (calorías, duración, frecuencia) |
| PC2 | ~18% | Masa corporal (peso, IMC) |
| PC3 | ~12% | Contraste estatura-peso (altura vs IMC) |
| PC4 | ~10% | Respuesta cardiovascular (BPM reposo y promedio) |

> **Nota:** 4 componentes retienen ~64% de la varianza. Para análisis de mayor precisión se recomiendan 6 componentes (~81.5%).

---

## 🗂️ Estructura del proyecto

```
📁 analisis-multivariado-gimnasio/
├── Analisis_multivariado_gimnasio.py   # Script principal
├── Miembros_gimnasio.xlsx              # Dataset de entrada
├── 1_histogramas_numericas.png
├── 1_barras_categoricas.png
├── 1_boxplots_calorias.png
├── 1_scatter_imc_grasa.png
├── 2_heatmap_correlacion.png
├── 3_scree_plot_pca.png
├── 3_heatmap_loadings.png
├── 3_biplot_pca.png
└── README.md
```

---

## 💡 Conceptos aplicados

- Estadísticas descriptivas: media, mediana, desviación estándar, coeficiente de variación, asimetría y curtosis
- Limpieza de datos: detección y eliminación de valores nulos, corrección de nombres de columnas
- Análisis de correlación de Pearson con visualización en heatmap
- Estandarización con `StandardScaler` (media 0, varianza 1) — requisito previo al PCA
- Reducción dimensional con PCA: varianza explicada, scree plot, loadings y biplot
- Visualización profesional con Matplotlib y Seaborn