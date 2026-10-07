# ==============================================================
# Evidencia de Aprendizaje — S35 Unidad 3
# Análisis Multivariado: Miembros Gimnasio
#
# Estudiantes: Hanna Jineth Contreras Salinas
# Programa   : Ingeniería de Software y Datos
#
# Descripción:
# Realizo un análisis exploratorio y descriptivo exhaustivo del
# conjunto de datos de miembros de un gimnasio (970 registros,
# 16 variables). Luego construyo la matriz de correlación con su
# representación gráfica, y finalmente aplico Análisis de
# Componentes Principales (PCA) para reducir las variables
# numéricas a 4 componentes conservando la máxima varianza.
# ==============================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ==============================================================
# PARÁMETROS GLOBALES
# ==============================================================

RAIZ          = Path(__file__).resolve().parent.parent
RUTA_DATOS    = RAIZ / "data" / "Miembros_gimnasio.xlsx"  # Archivo de entrada
DIR_FIGURAS   = RAIZ / "figures"                          # Carpeta de salida
# Con --mostrar cada gráfica se abre en una ventana además de guardarse
MOSTRAR_GRAFICAS = "--mostrar" in sys.argv
N_COMPONENTES = 4                          # Componentes PCA objetivo
SEMILLA       = 42                         # Semilla para reproducibilidad
COLOR_PPAL    = "#2563EB"                  # Azul corporativo para gráficas
COLOR_ACENTO  = "#F59E0B"                  # Ámbar de acento
PALETTE_CAT   = "Set2"                     # Paleta para variables categóricas


# ==============================================================
# FUNCIÓN AUXILIAR — Separador visual en consola
# ==============================================================

def guardar_figura(nombre: str) -> None:
    """Guardo la figura actual en figures/ y la muestro si se pidió --mostrar."""
    DIR_FIGURAS.mkdir(exist_ok=True)
    plt.savefig(DIR_FIGURAS / nombre, dpi=150, bbox_inches="tight")
    if MOSTRAR_GRAFICAS:
        plt.show(block=False)
        plt.pause(2)
    plt.close()


def separador(titulo: str) -> None:
    """Imprimo un separador visual con el título del numeral."""
    ancho = 60
    print("\n" + "=" * ancho)
    print(f"  {titulo}")
    print("=" * ancho)


# ==============================================================
# CARGA Y LIMPIEZA DE DATOS
# ==============================================================

def cargar_datos(ruta: Path) -> pd.DataFrame:
    """
    Cargo el archivo Excel y realizo una limpieza mínima:
    elimino espacios en los nombres de columnas, corrijo el nombre
    de la columna de BPM en reposo y descarto las 9 filas cuya
    categoría IMC está vacía, ya que representan menos del 1 %.
    """
    df = pd.read_excel(ruta)

    # Elimino espacios accidentales en los encabezados.
    df.columns = df.columns.str.strip()

    # El nombre original tiene un typo ("Reposo_BPMBPM"); lo corrijo.
    df.rename(columns={"Reposo_BPMBPM": "Reposo_BPM"}, inplace=True)

    # Descarto filas con Categoría_IMC nula (solo 9 de 970).
    filas_antes = len(df)
    df.dropna(subset=["Categoría_IMC"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    filas_despues = len(df)

    print(f"\n[DATOS] Registros cargados : {filas_antes}")
    print(f"[DATOS] Registros eliminados: {filas_antes - filas_despues} (Categoría_IMC nula)")
    print(f"[DATOS] Registros finales  : {filas_despues}")
    print(f"[DATOS] Variables          : {df.shape[1]}")

    return df


# ==============================================================
# NUMERAL 1 — Análisis descriptivo y exploratorio exhaustivo
# ==============================================================

def analisis_descriptivo(df: pd.DataFrame) -> None:
    """
    Calculo medidas de tendencia central y variabilidad para todas
    las variables numéricas, presento distribuciones individuales,
    cruces entre variables categóricas y numéricas, y extraigo
    conclusiones clave sobre los patrones del conjunto de datos.
    """
    separador("NUMERAL 1 — Análisis Descriptivo y Exploratorio")

    # ----------------------------------------------------------
    # 1.1 Selecciono las columnas numéricas para el resumen
    # ----------------------------------------------------------
    numericas = df.select_dtypes(include="number").columns.tolist()

    resumen = df[numericas].describe().T
    resumen["rango"]    = resumen["max"] - resumen["min"]
    resumen["cv (%)"]   = (resumen["std"] / resumen["mean"] * 100).round(2)
    resumen["mediana"]  = df[numericas].median()
    resumen["asimetría"]= df[numericas].skew().round(4)
    resumen["curtosis"] = df[numericas].kurt().round(4)

    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", 12)
    pd.set_option("display.width", 120)

    print("\n--- Resumen estadístico completo (variables numéricas) ---")
    print(resumen[["count","mean","mediana","std","min","max","rango","cv (%)","asimetría","curtosis"]])

    # ----------------------------------------------------------
    # Interpretación del resumen
    # ----------------------------------------------------------
    print("""
Interpretación:
  • La edad promedio de los miembros es 38.7 años (rango 18–59), con un
    coeficiente de variación (CV) del 31.4 %, lo que indica dispersión moderada.
  • El peso medio es 73.7 kg y la altura 1.72 m, resultando en un IMC promedio
    de 24.9, en el límite superior del rango normal (≤24.9).
  • Las frecuencias cardíacas (Máx_BPM ≈ 180, Promedio ≈ 144, Reposo ≈ 62)
    corresponden a perfiles de ejercicio de intensidad media-alta.
  • Las Calorías_quemadas presentan el CV más alto entre las variables de
    entrenamiento (30.2 %; solo la Edad lo supera con 31.4 %), lo que refleja
    alta variabilidad entre sesiones y tipos de entrenamiento.
  • El Porcentaje_grasa tiene asimetría NEGATIVA (-0.63), indicando que la
    mayoría de los miembros tiene porcentaje de grasa medio-alto, con pocos
    casos de grasa muy baja.
  • El IMC presenta asimetría positiva (0.78): hay una cola hacia valores
    altos, es decir, más miembros con sobrepeso u obesidad que con IMC muy bajo.
""")

    # ----------------------------------------------------------
    # 1.2 Gráfica 1 — Histogramas de todas las variables numéricas
    # ----------------------------------------------------------
    fig, axes = plt.subplots(4, 3, figsize=(16, 14))
    axes = axes.flatten()

    for i, col in enumerate(numericas):
        axes[i].hist(df[col], bins=25, color=COLOR_PPAL, edgecolor="white", alpha=0.85)
        media_val = df[col].mean()
        axes[i].axvline(media_val, color=COLOR_ACENTO, linestyle="--", linewidth=1.5, label=f"Media: {media_val:.1f}")
        axes[i].set_title(col, fontsize=10, fontweight="bold")
        axes[i].set_xlabel("Valor", fontsize=8)
        axes[i].set_ylabel("Frecuencia", fontsize=8)
        axes[i].legend(fontsize=7)
        axes[i].grid(axis="y", linestyle="--", alpha=0.5)

    # Oculto los ejes sobrantes si hay menos variables que celdas
    for j in range(len(numericas), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Distribuciones de Variables Numéricas — Miembros Gimnasio",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    guardar_figura("1_histogramas_numericas.png")
    print("\n[Gráfica guardada] 1_histogramas_numericas.png")

    # ----------------------------------------------------------
    # 1.3 Gráfica 2 — Distribución de variables categóricas
    # ----------------------------------------------------------
    categoricas = ["Género", "Tipo_entrenamiento", "Nivel_experiencia", "Categoría_IMC"]
    colores_cat = sns.color_palette(PALETTE_CAT, 6)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    for i, col in enumerate(categoricas):
        conteos = df[col].value_counts()
        bars = axes[i].bar(conteos.index, conteos.values,
                           color=colores_cat[:len(conteos)], edgecolor="white")
        # Anoto el conteo y el porcentaje sobre cada barra
        total = conteos.sum()
        for bar, val in zip(bars, conteos.values):
            axes[i].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 5,
                         f"{val}\n({val/total*100:.1f}%)",
                         ha="center", va="bottom", fontsize=8.5)
        axes[i].set_title(f"Distribución: {col}", fontsize=11, fontweight="bold")
        axes[i].set_ylabel("Cantidad de miembros", fontsize=9)
        axes[i].set_xlabel(col, fontsize=9)
        axes[i].grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle("Distribución de Variables Categóricas — Miembros Gimnasio",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    guardar_figura("1_barras_categoricas.png")
    print("[Gráfica guardada] 1_barras_categoricas.png")

    # ----------------------------------------------------------
    # 1.4 Gráfica 3 — Boxplots: Calorías quemadas por Tipo de entrenamiento
    # ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    df.boxplot(column="Calorías_quemadas", by="Tipo_entrenamiento",
               ax=axes[0], patch_artist=True,
               boxprops=dict(facecolor=COLOR_PPAL, alpha=0.7),
               medianprops=dict(color=COLOR_ACENTO, linewidth=2))
    axes[0].set_title("Calorías quemadas por Tipo de entrenamiento", fontsize=10, fontweight="bold")
    axes[0].set_xlabel("Tipo de entrenamiento"); axes[0].set_ylabel("Calorías quemadas")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)
    plt.sca(axes[0]); plt.title("")

    df.boxplot(column="Calorías_quemadas", by="Nivel_experiencia",
               ax=axes[1], patch_artist=True,
               boxprops=dict(facecolor="#10B981", alpha=0.7),
               medianprops=dict(color=COLOR_ACENTO, linewidth=2))
    axes[1].set_title("Calorías quemadas por Nivel de experiencia", fontsize=10, fontweight="bold")
    axes[1].set_xlabel("Nivel de experiencia"); axes[1].set_ylabel("Calorías quemadas")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)
    plt.sca(axes[1]); plt.title("")

    fig.suptitle("Calorías quemadas según tipo de entrenamiento y nivel de experiencia",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    guardar_figura("1_boxplots_calorias.png")
    print("[Gráfica guardada] 1_boxplots_calorias.png")

    # ----------------------------------------------------------
    # 1.5 Gráfica 4 — Scatter: IMC vs Porcentaje de grasa, coloreado por Género
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    colores_genero = {"Masculino": COLOR_PPAL, "Femenino": "#EF4444"}

    for genero, grupo in df.groupby("Género"):
        ax.scatter(grupo["IMC"], grupo["Porcentaje_grasa"],
                   alpha=0.45, s=22, label=genero, color=colores_genero[genero])

    ax.set_xlabel("IMC", fontsize=11)
    ax.set_ylabel("Porcentaje de grasa corporal (%)", fontsize=11)
    ax.set_title("IMC vs Porcentaje de grasa — por Género", fontsize=13, fontweight="bold")
    ax.legend(title="Género", fontsize=10)
    ax.grid(linestyle="--", alpha=0.4)

    plt.tight_layout()
    guardar_figura("1_scatter_imc_grasa.png")
    print("[Gráfica guardada] 1_scatter_imc_grasa.png")

    # ----------------------------------------------------------
    # 1.6 Cruce: Estadísticas de IMC por Nivel de experiencia y Género
    # ----------------------------------------------------------
    print("\n--- Cruce: IMC promedio por Nivel_experiencia y Género ---")
    cruce = df.pivot_table(values="IMC", index="Nivel_experiencia",
                           columns="Género", aggfunc=["mean","std"]).round(2)
    print(cruce)

    print("\n--- Cruce: Calorías promedio por Tipo_entrenamiento y Nivel_experiencia ---")
    cruce2 = df.pivot_table(values="Calorías_quemadas",
                            index="Tipo_entrenamiento",
                            columns="Nivel_experiencia",
                            aggfunc="mean").round(0)
    print(cruce2)

    print("""
Conclusiones clave del análisis exploratorio:
  • El IMC promedio es consistentemente mayor en hombres que en mujeres
    en todos los niveles de experiencia (Alto: 26.6 vs 22.8, Bajo: 26.6 vs 22.4,
    Medio: 27.2 vs 23.0), lo que indica diferencias reales de composición
    corporal por género, independientemente de la experiencia.
  • Los entrenamientos de tipo HIIT y Strength generan más calorías que Yoga
    en todos los niveles; sin embargo, el nivel de experiencia es el factor
    dominante: nivel Alto quema en promedio ≈1 268 cal vs ≈715 cal en nivel Bajo,
    independientemente del tipo de ejercicio.
  • Los miembros con mayor nivel de experiencia tienen IMC y porcentajes de grasa
    menores, lo que es consistente con la acumulación de hábitos saludables.
  • La distribución del IMC presenta asimetría positiva: hay más miembros con
    IMC elevado (sobrepeso/obesidad) que con IMC muy bajo, lo que representa
    una oportunidad para programas de intervención nutricional.
    """)


# ==============================================================
# NUMERAL 2 — Matriz de correlación
# ==============================================================

def matriz_correlacion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculo la matriz de correlación de Pearson para todas las
    variables numéricas, la imprimo en consola y la visualizo con
    un heatmap con anotaciones para facilitar la lectura.
    Retorno la matriz para reutilizarla en el PCA si se requiere.
    """
    separador("NUMERAL 2 — Matriz de Correlación")

    numericas = df.select_dtypes(include="number")

    # Calculo la correlación de Pearson entre pares de variables numéricas.
    corr = numericas.corr(method="pearson")

    print("\n--- Matriz de correlación de Pearson ---")
    print(corr.round(3))

    # ----------------------------------------------------------
    # Identifico las correlaciones más fuertes (|r| > 0.4)
    # ----------------------------------------------------------
    print("\n--- Pares con correlación |r| > 0.40 ---")
    pares_fuertes = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) > 0.40:
                pares_fuertes.append((cols[i], cols[j], round(r, 4)))

    pares_fuertes.sort(key=lambda x: abs(x[2]), reverse=True)
    for v1, v2, r in pares_fuertes:
        signo = "positiva" if r > 0 else "negativa"
        print(f"  {v1:42s} ↔ {v2:42s}  r = {r:+.4f}  ({signo})")

    # ----------------------------------------------------------
    # Heatmap de la matriz de correlación
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 10))

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.5,
        linecolor="white",
        square=True,
        ax=ax,
        annot_kws={"size": 8},
    )

    ax.set_title("Matriz de Correlación de Pearson — Variables Numéricas\nMiembros Gimnasio",
                 fontsize=13, fontweight="bold", pad=14)
    ax.tick_params(axis="x", rotation=45, labelsize=8.5)
    ax.tick_params(axis="y", rotation=0,  labelsize=8.5)

    plt.tight_layout()
    guardar_figura("2_heatmap_correlacion.png")
    print("\n[Gráfica guardada] 2_heatmap_correlacion.png")

    print("""
Conclusiones de la matriz de correlación:
  • Duración_sesión y Calorías_quemadas: r = +0.909 (muy fuerte positiva).
    La correlación más alta del dataset: sesiones más largas queman
    considerablemente más calorías.
  • Peso e IMC: r = +0.854 (muy fuerte positiva). Esperada dado que el IMC
    se calcula directamente a partir del peso y la altura.
  • Duración_sesión y Frecuencia_entrenamiento: r = +0.641 (fuerte positiva).
    Miembros que entrenan más días también tienen sesiones más largas.
  • Calorías_quemadas y Porcentaje_grasa: r = -0.598 (fuerte negativa).
    Mayor actividad sostenida se asocia a menor porcentaje de grasa.
  • Porcentaje_grasa e Ingesta_agua: r = -0.588 (fuerte negativa).
    Mayor hidratación diaria se asocia a menor grasa corporal.
  • Duración_sesión y Porcentaje_grasa: r = -0.580 (fuerte negativa).
    Sesiones más largas se asocian a menor porcentaje de grasa.
  • Calorías_quemadas y Frecuencia_entrenamiento: r = +0.576 (moderada positiva).
    Entrenar más días a la semana resulta en más calorías quemadas totales.
  • Porcentaje_grasa y Frecuencia_entrenamiento: r = -0.536 (moderada negativa).
    Mayor frecuencia semanal se asocia a menor grasa corporal.
  • Las BPM (Max, Promedio, Reposo) tienen correlaciones muy bajas con el resto
    (|r| < 0.10), confirmando que miden dimensiones cardiovasculares independientes.
""")

    return corr


# ==============================================================
# NUMERAL 3 — Análisis de Componentes Principales (PCA)
# ==============================================================

def analisis_pca(df: pd.DataFrame, n_comp: int = N_COMPONENTES) -> None:
    """
    Aplico PCA para reducir las variables numéricas del dataset a
    n_comp componentes principales. Estandarizo previamente los
    datos (media 0, desviación 1) para evitar que variables con
    distintas escalas dominen el análisis. Evalúo la varianza
    explicada, los loadings de cada componente e interpreto si la
    reducción es adecuada para este conjunto de variables.
    """
    separador("NUMERAL 3 — Análisis de Componentes Principales (PCA)")

    # ----------------------------------------------------------
    # 3.1 Preparación: selecciono variables numéricas y estandarizo
    # ----------------------------------------------------------
    numericas = df.select_dtypes(include="number")
    nombres_vars = numericas.columns.tolist()

    # Estandarizo con StandardScaler: esto es obligatorio en PCA
    # para que todas las variables tengan igual peso inicial.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numericas)

    print(f"\nVariables incluidas en el PCA ({len(nombres_vars)} en total):")
    for v in nombres_vars:
        print(f"  • {v}")

    # ----------------------------------------------------------
    # 3.2 PCA completo primero — para ver la varianza total acumulada
    # ----------------------------------------------------------
    pca_completo = PCA(random_state=SEMILLA)
    pca_completo.fit(X_scaled)

    varianza_exp      = pca_completo.explained_variance_ratio_
    varianza_acum     = np.cumsum(varianza_exp)
    eigenvalues       = pca_completo.explained_variance_

    print("\n--- Varianza explicada por componente (PCA completo) ---")
    print(f"{'Componente':<14} {'Varianza %':>12} {'Varianza Acum %':>16} {'Eigenvalue':>12}")
    for i, (ve, va, ev) in enumerate(zip(varianza_exp, varianza_acum, eigenvalues), 1):
        marca = " ◄ (seleccionada)" if i <= n_comp else ""
        print(f"  PC{i:<11} {ve*100:>11.2f}% {va*100:>15.2f}% {ev:>12.4f}{marca}")

    varianza_retenida = varianza_acum[n_comp - 1]
    print(f"\n  Varianza retenida con {n_comp} componentes: {varianza_retenida*100:.2f}%")

    # ----------------------------------------------------------
    # 3.3 Gráfica 5 — Scree plot + varianza acumulada
    # ----------------------------------------------------------
    componentes_idx = np.arange(1, len(varianza_exp) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scree plot
    axes[0].bar(componentes_idx, varianza_exp * 100,
                color=COLOR_PPAL, edgecolor="white", alpha=0.85, label="Varianza individual")
    axes[0].plot(componentes_idx, varianza_exp * 100,
                 marker="o", color=COLOR_ACENTO, linewidth=2, markersize=6)
    axes[0].axvline(n_comp + 0.5, color="red", linestyle="--", linewidth=1.5,
                    label=f"Corte en PC{n_comp}")
    axes[0].set_xlabel("Componente Principal", fontsize=10)
    axes[0].set_ylabel("Varianza explicada (%)", fontsize=10)
    axes[0].set_title("Scree Plot — Varianza por Componente", fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    # Varianza acumulada
    axes[1].plot(componentes_idx, varianza_acum * 100,
                 marker="o", color=COLOR_PPAL, linewidth=2.5, markersize=7)
    axes[1].axhline(varianza_retenida * 100, color=COLOR_ACENTO, linestyle="--",
                    linewidth=1.5, label=f"PC{n_comp}: {varianza_retenida*100:.1f}%")
    axes[1].axhline(80, color="gray", linestyle=":", linewidth=1.2, label="Umbral 80%")
    axes[1].fill_between(componentes_idx[:n_comp],
                         varianza_acum[:n_comp] * 100,
                         alpha=0.15, color=COLOR_PPAL)
    axes[1].set_xlabel("Número de componentes", fontsize=10)
    axes[1].set_ylabel("Varianza acumulada (%)", fontsize=10)
    axes[1].set_title("Varianza Acumulada — PCA", fontsize=11, fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].grid(linestyle="--", alpha=0.4)
    axes[1].set_ylim(0, 105)

    fig.suptitle("Análisis de Componentes Principales — Miembros Gimnasio",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    guardar_figura("3_scree_plot_pca.png")
    print("\n[Gráfica guardada] 3_scree_plot_pca.png")

    # ----------------------------------------------------------
    # 3.4 PCA con n_comp componentes — Loadings (contribuciones)
    # ----------------------------------------------------------
    pca = PCA(n_components=n_comp, random_state=SEMILLA)
    X_pca = pca.fit_transform(X_scaled)

    # Construyo el DataFrame de loadings para visualizarlo ordenadamente.
    loadings = pd.DataFrame(
        pca.components_.T,
        index=nombres_vars,
        columns=[f"PC{i+1}" for i in range(n_comp)]
    )

    print(f"\n--- Loadings (contribuciones) de cada variable en las {n_comp} componentes ---")
    print(loadings.round(4))

    # ----------------------------------------------------------
    # 3.5 Gráfica 6 — Heatmap de loadings
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        loadings,
        annot=True,
        fmt=".3f",
        cmap="RdBu_r",
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        annot_kws={"size": 9},
    )
    ax.set_title(f"Loadings de las {n_comp} Componentes Principales\n(contribución de cada variable)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Componente Principal", fontsize=10)
    ax.set_ylabel("Variable original", fontsize=10)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    plt.tight_layout()
    guardar_figura("3_heatmap_loadings.png")
    print("[Gráfica guardada] 3_heatmap_loadings.png")

    # ----------------------------------------------------------
    # 3.6 Gráfica 7 — Biplot PC1 vs PC2 coloreado por IMC
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 7))

    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                         c=df["IMC"], cmap="YlOrRd", alpha=0.55, s=20, edgecolors="none")
    plt.colorbar(scatter, ax=ax, label="IMC")

    # Dibujo los vectores de los loadings (flechas)
    escala = 3.5
    for i, var in enumerate(nombres_vars):
        ax.annotate("", xy=(loadings.iloc[i, 0] * escala, loadings.iloc[i, 1] * escala),
                    xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="#1E3A5F", lw=1.5))
        ax.text(loadings.iloc[i, 0] * escala * 1.08,
                loadings.iloc[i, 1] * escala * 1.08,
                var, fontsize=7.5, color="#1E3A5F", ha="center")

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel(f"PC1 ({varianza_exp[0]*100:.1f}% varianza)", fontsize=11)
    ax.set_ylabel(f"PC2 ({varianza_exp[1]*100:.1f}% varianza)", fontsize=11)
    ax.set_title("Biplot PCA — PC1 vs PC2 (coloreado por IMC)", fontsize=12, fontweight="bold")
    ax.grid(linestyle="--", alpha=0.3)

    plt.tight_layout()
    guardar_figura("3_biplot_pca.png")
    print("[Gráfica guardada] 3_biplot_pca.png")

    # ----------------------------------------------------------
    # 3.7 Gráfica 8 — Scatter PC1 vs PC2 coloreado por Tipo de entrenamiento
    # ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    tipos    = df["Tipo_entrenamiento"].unique()
    paleta   = sns.color_palette(PALETTE_CAT, len(tipos))

    for tipo, color in zip(tipos, paleta):
        mask = df["Tipo_entrenamiento"] == tipo
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   label=tipo, color=color, alpha=0.55, s=20, edgecolors="none")

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel(f"PC1 ({varianza_exp[0]*100:.1f}% varianza)", fontsize=11)
    ax.set_ylabel(f"PC2 ({varianza_exp[1]*100:.1f}% varianza)", fontsize=11)
    ax.set_title("PC1 vs PC2 — por Tipo de entrenamiento", fontsize=12, fontweight="bold")
    ax.legend(title="Tipo entrenamiento", fontsize=9, markerscale=2)
    ax.grid(linestyle="--", alpha=0.3)

    plt.tight_layout()
    guardar_figura("3_scatter_pca_tipo.png")
    print("[Gráfica guardada] 3_scatter_pca_tipo.png")

    # ----------------------------------------------------------
    # 3.8 Resumen final e interpretación del PCA
    # ----------------------------------------------------------
    print(f"""
--- Resumen PCA: {N_COMPONENTES} componentes seleccionadas ---
  Varianza retenida  : {varianza_retenida*100:.2f}%
  Varianza descartada: {(1 - varianza_retenida)*100:.2f}%

Interpretación de las componentes:
  • PC1 ({varianza_exp[0]*100:.1f}%): Cargas altas en Calorías_quemadas (0.478),
    Duración_sesión (0.453) y Frecuencia_entrenamiento (0.386), con carga
    negativa en Porcentaje_grasa (-0.458). Captura el VOLUMEN DE ENTRENAMIENTO:
    miembros más activos y con menor grasa tienen valores altos en PC1.

  • PC2 ({varianza_exp[1]*100:.1f}%): Cargas altas en Peso (0.634) e IMC (0.545).
    Captura la MASA CORPORAL: miembros más pesados con mayor IMC tienen
    valores altos en PC2.

  • PC3 ({varianza_exp[2]*100:.1f}%): Carga alta en Altura (0.694) y negativa
    en IMC (-0.496). Captura el CONTRASTE ESTATURA-PESO: personas altas con
    bajo IMC tienen valores altos en PC3.

  • PC4 ({varianza_exp[3]*100:.1f}%): Cargas altas en Promedio_BPM (0.798) y
    Reposo_BPM (0.460). Captura la RESPUESTA CARDIOVASCULAR: miembros con
    frecuencia cardíaca alta en reposo y durante el ejercicio.

¿Es adecuada la reducción de variables?
  La reducción a 4 componentes es técnicamente posible pero debe evaluarse con
  criterio crítico:
  1. Se retiene el {varianza_retenida*100:.1f}% de la información original.
     Este valor es inferior al umbral convencional del 80%, lo que implica que
     se pierde aproximadamente un 36% de la variabilidad del dataset original.
     Si se requiere mayor fidelidad, sería recomendable usar 6 componentes,
     que retienen el 81.5%.
  2. El Scree Plot no muestra un codo pronunciado único; la varianza disminuye
     gradualmente hasta PC6, lo que indica que las 12 variables numéricas
     aportan dimensiones de información relativamente independientes entre sí.
  3. El heatmap de loadings confirma que variables con alta correlación real
     (Peso-IMC, Duración-Calorías) quedan agrupadas en la misma componente,
     validando que el PCA captura correctamente las redundancias del dataset.
  4. Los tipos de entrenamiento no forman clústeres separados en el espacio PCA,
     lo que indica que el tipo de ejercicio no es el factor diferenciador principal;
     los parámetros corporales dependen más de la experiencia y la constancia.
  Conclusión: Para el objetivo de reducir a exactamente 4 componentes conservando
  el máximo de información, el PCA es la herramienta correcta. Sin embargo, con
  este dataset la reducción a 4 componentes no es ideal (64.3% de varianza);
  se recomienda al menos 6 componentes (81.5%) para análisis posteriores
  que requieran mayor precisión.
""")


# ==============================================================
# PUNTO DE ENTRADA
# ==============================================================

if __name__ == "__main__":

    # Cargo el dataset una sola vez y lo comparto entre los tres numerales.
    df = cargar_datos(RUTA_DATOS)

    # Ejecuto los tres numerales de forma secuencial.
    analisis_descriptivo(df)
    corr = matriz_correlacion(df)
    analisis_pca(df, n_comp=N_COMPONENTES)

    print("\n" + "=" * 60)
    print("  Análisis completado. Todas las gráficas han sido")
    print(f"  guardadas como archivos PNG en {DIR_FIGURAS}.")
    print("=" * 60)
