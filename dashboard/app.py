from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(
    page_title="Segmentación K-Means | Steam Real (Kaggle)",
    layout="wide",
    initial_sidebar_state="expanded",
)

sns.set_theme(style="whitegrid")

PALETA_CLUSTERS = {
    "Económicos de Baja Aceptación": "#ef4444",
    "Económicos Bien Valorados": "#3b82f6",
    "Gama Alta / Premium": "#f59e0b",
    "Éxitos de Alta Retención": "#10b981",
}

ORDEN_CLUSTERS = [
    "Económicos de Baja Aceptación",
    "Económicos Bien Valorados",
    "Gama Alta / Premium",
    "Éxitos de Alta Retención",
]

ETIQUETAS_VARS = {
    "price": "Precio (GBP)",
    "calificacion_pct": "Calificación Positiva (%)",
    "horas_promedio": "Horas de Juego Promedio",
    "positive_ratings": "Reseñas Positivas",
    "propietarios_est": "Propietarios Estimados",
}

DESCRIPCIONES_GRUPOS = {
    "Económicos de Baja Aceptación": (
        "Juegos baratos o gratuitos (promedio 3.33 GBP) con recepción desfavorable (50.90% de reseñas positivas), "
        "poca retención (3.15 horas) y menor alcance comercial promedio (~210 mil propietarios estimados)."
    ),
    "Económicos Bien Valorados": (
        "Es el corazón del catálogo de Steam (52.71% de los juegos activos): precios accesibles (4.97 GBP promedio), "
        "la calificación más alta del mercado (83.41%) y sesiones cortas o medias (3.77 horas promedio), como Portal 2 o Undertale."
    ),
    "Gama Alta / Premium": (
        "Lanzamientos de precio alto en Steam (20.58 GBP promedio) con sólida valoración (81.25%), "
        "retención moderada (6.84 horas promedio) y amplia comunidad (~545 mil propietarios), como DOOM o BioShock Infinite."
    ),
    "Éxitos de Alta Retención": (
        "El grupo menos numeroso (5.58%) pero el de mayor impacto en horas y alcance: precio medio-alto (16.20 GBP), "
        "buena nota (79.46%), una retención masiva de 33.52 horas promedio y casi 1.49 millones de propietarios estimados (ej. The Witcher 3, Borderlands 2)."
    ),
}


@st.cache_data
def cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    raiz = Path(__file__).resolve().parent.parent
    df = pd.read_csv(raiz / "datos" / "steam_procesado.csv")
    centroides = pd.read_csv(raiz / "datos" / "centroides_kmeans.csv")
    eval_k = pd.read_csv(raiz / "datos" / "metricas_evaluacion_k.csv")
    resumen = json.loads((raiz / "datos" / "resumen_modelo.json").read_text(encoding="utf-8"))
    return df, centroides, eval_k, resumen


def main() -> None:
    st.markdown(
        """
        <style>
            .block-container { padding-top: 1.0rem; padding-bottom: 0.8rem; }
            h1 { font-size: 1.65rem !important; margin-bottom: 0.15rem !important; }
            h2 { font-size: 1.25rem !important; margin-top: 0.1rem !important; margin-bottom: 0.3rem !important; }
            h3 { font-size: 1.02rem !important; margin-top: 0.15rem !important; margin-bottom: 0.2rem !important; }
            hr { margin-top: 0.4rem !important; margin-bottom: 0.4rem !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    df, centroides, eval_k, resumen = cargar_datos()

    st.title("Dashboard Interactivo: Segmentación de Videojuegos Reales en Steam (Kaggle)")
    st.markdown(
        f"Análisis exploratorio y agrupamiento no supervisado (**K = {resumen['k_optimo']}**) sobre "
        f"**{resumen['registros_limpios']:,} videojuegos reales limpios** (de `{resumen['registros_iniciales']:,}` originales en `steam.csv`)."
    )

    pagina_param = st.query_params.get("page", "general")
    indice_default = 1 if pagina_param == "segmentacion" else 0

    st.sidebar.header("Navegación")
    pagina = st.sidebar.radio(
        "Selecciona la página:",
        ["Página 1. Análisis general", "Página 2. Segmentación mediante K-Means"],
        index=indice_default,
    )

    st.sidebar.divider()
    st.sidebar.header("Filtros Interactivos")

    # Filtro 1: Segmento / Cluster
    segmentos_disponibles = [s for s in ORDEN_CLUSTERS if s in df["segmento"].unique()]
    segmentos_sel = st.sidebar.multiselect(
        "1. Segmento / Clúster:",
        options=segmentos_disponibles,
        default=segmentos_disponibles,
    )

    # Filtro 2: Genero principal
    generos_disponibles = sorted(df["genero_principal"].unique().tolist())
    generos_sel = st.sidebar.multiselect(
        "2. Género Principal:",
        options=generos_disponibles,
        default=generos_disponibles,
    )

    # Filtro 3: Rango de Precio GBP
    precio_min = float(df["price"].min())
    precio_max = float(df["price"].max())
    rango_precio = st.sidebar.slider(
        "3. Rango de Precio (GBP):",
        min_value=round(precio_min, 2),
        max_value=round(precio_max, 2),
        value=(round(precio_min, 2), round(precio_max, 2)),
    )

    df_filtrado = df[
        df["segmento"].isin(segmentos_sel)
        & df["genero_principal"].isin(generos_sel)
        & df["price"].between(rango_precio[0], rango_precio[1])
    ].copy()

    if df_filtrado.empty:
        st.warning("No hay videojuegos que coincidan con los filtros seleccionados. Ajusta el panel lateral.")
        return

    st.sidebar.divider()
    st.sidebar.info(f"Registros visibles: **{len(df_filtrado):,}** de **{len(df):,}**")

    # =========================================================================
    # PAGINA 1: ANALISIS GENERAL
    # =========================================================================
    if pagina == "Página 1. Análisis general":
        st.header("Página 1. Análisis General del Catálogo Real de Steam")

        # 4 Indicadores KPI relevantes
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Videojuegos", f"{len(df_filtrado):,}", f"{len(df_filtrado)/len(df)*100:.1f}% del catálogo limpio")
        k2.metric("Precio Promedio", f"{df_filtrado['price'].mean():.2f} GBP", f"Mediana: {df_filtrado['price'].median():.2f} GBP")
        k3.metric("Calificación Promedio", f"{df_filtrado['calificacion_pct'].mean():.1f}%", f"Máx: {df_filtrado['calificacion_pct'].max():.1f}%")
        k4.metric("Horas Promedio", f"{df_filtrado['horas_promedio'].mean():.2f} h", f"Reseñas med: {df_filtrado['positive_ratings'].median():,.0f}")

        st.divider()

        col_izq, col_der = st.columns(2)
        with col_izq:
            st.subheader("Top 10 Géneros Principales en Steam")
            fig_gen, ax_gen = plt.subplots(figsize=(7, 3.3))
            orden_g = df_filtrado["genero_principal"].value_counts().head(10).index
            df_top_g = df_filtrado[df_filtrado["genero_principal"].isin(orden_g)]
            sns.countplot(data=df_top_g, y="genero_principal", order=orden_g, hue="genero_principal", palette="viridis", legend=False, ax=ax_gen)
            ax_gen.set_xlabel("Cantidad de Videojuegos")
            ax_gen.set_ylabel("Género Principal")
            for c in ax_gen.containers:
                ax_gen.bar_label(c, padding=3, fontsize=8.5)
            plt.tight_layout()
            st.pyplot(fig_gen)
            plt.close(fig_gen)

        with col_der:
            st.subheader("Matriz de Correlación de Variables Numéricas")
            cols_num = ["anio_lanzamiento", "price", "calificacion_pct", "horas_promedio", "positive_ratings"]
            nombres_corr = {
                "anio_lanzamiento": "Año",
                "price": "Precio",
                "calificacion_pct": "Calif. (%)",
                "horas_promedio": "Horas Prom.",
                "positive_ratings": "Reseñas Pos.",
            }
            fig_corr, ax_corr = plt.subplots(figsize=(7, 3.3))
            corr = df_filtrado[cols_num].corr()
            corr.index = [nombres_corr[c] for c in cols_num]
            corr.columns = [nombres_corr[c] for c in cols_num]
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax_corr)
            plt.xticks(rotation=20, ha="right", fontsize=9)
            plt.yticks(rotation=0, fontsize=9)
            plt.tight_layout()
            st.pyplot(fig_corr)
            plt.close(fig_corr)

        col_h1, col_h2 = st.columns(2)
        with col_h1:
            st.subheader("Distribución del Precio (GBP)")
            fig_p, ax_p = plt.subplots(figsize=(7, 2.9))
            sns.histplot(df_filtrado["price"], bins=25, kde=True, color="#3b82f6", ax=ax_p)
            ax_p.set_xlabel("Precio (GBP)")
            ax_p.set_ylabel("Frecuencia")
            plt.tight_layout()
            st.pyplot(fig_p)
            plt.close(fig_p)

        with col_h2:
            st.subheader("Distribución de Horas de Juego Promedio")
            fig_hrs, ax_hrs = plt.subplots(figsize=(7, 2.9))
            sns.histplot(df_filtrado["horas_promedio"], bins=25, kde=True, color="#10b981", ax=ax_hrs)
            ax_hrs.set_xlabel("Horas Promedio de Juego")
            ax_hrs.set_ylabel("Frecuencia")
            plt.tight_layout()
            st.pyplot(fig_hrs)
            plt.close(fig_hrs)

        st.subheader("Estadísticas Descriptivas Generales del Dataset Filtrado")
        desc_general = df_filtrado[
            ["anio_lanzamiento", "price", "calificacion_pct", "horas_promedio", "positive_ratings", "negative_ratings", "propietarios_est"]
        ].describe().T.round(2)
        st.dataframe(desc_general, width="stretch")

    # =========================================================================
    # PAGINA 2: SEGMENTACION MEDIANTE K-MEANS
    # =========================================================================
    else:
        st.header("Página 2. Segmentación mediante K-Means (K = 4)")

        cluster_dom = df_filtrado["segmento"].value_counts().idxmax()
        prop_prom = df_filtrado["propietarios_est"].mean() / 1000.0
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Clústeres Óptimos (K)", f"{resumen['k_optimo']}", f"Iteraciones: {resumen['iteraciones_k4']}")
        k2.metric("Silhouette Score", f"{resumen['silhouette_k4']:.4f}", f"Inercia: {resumen['inercia_k4']:,.1f}")
        k3.metric("Segmento Predominante", cluster_dom.replace("Económicos ", "Econ. "), f"{df_filtrado['segmento'].value_counts().max():,} títulos")
        k4.metric("Propietarios Est. Prom.", f"{prop_prom:,.1f} mil", "Promedio en selección")

        st.divider()

        vista_param = st.query_params.get("vista", "agrupamientos")
        vista_idx = 1 if vista_param == "comparativas" else 0
        vista_p2 = st.radio(
            "Sección de resultados:",
            ["Agrupamientos y Distribución", "Comparativas, Estadísticas y Hallazgos"],
            index=vista_idx,
            horizontal=True,
        )

        if vista_p2 == "Agrupamientos y Distribución":
            col_g1, col_g2 = st.columns([1, 1.35])

            with col_g1:
                st.subheader("Distribución de Registros por Clúster")
                conteo = df_filtrado["segmento"].value_counts().reindex([s for s in ORDEN_CLUSTERS if s in df_filtrado["segmento"].unique()])
                fig_dist, ax_dist = plt.subplots(figsize=(6.5, 3.1))
                colores_bar = [PALETA_CLUSTERS[s] for s in conteo.index]
                bars = ax_dist.barh(range(len(conteo)), conteo.values, color=colores_bar)
                ax_dist.set_yticks(range(len(conteo)))
                ax_dist.set_yticklabels(conteo.index, fontsize=9)
                ax_dist.invert_yaxis()
                ax_dist.set_xlabel("Cantidad de Videojuegos")
                total_vis = len(df_filtrado)
                for bar, val in zip(bars, conteo.values):
                    ax_dist.text(
                        bar.get_width() + max(conteo.values) * 0.02,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:,} ({val / total_vis * 100:.1f}%)",
                        va="center",
                        fontsize=8.5,
                        fontweight="bold",
                    )
                ax_dist.set_xlim(0, max(conteo.values) * 1.40)
                plt.tight_layout()
                st.pyplot(fig_dist)
                plt.close(fig_dist)

            with col_g2:
                st.subheader("Gráfica de Dispersión de Agrupamientos y Centroides")
                c_eje1, c_eje2 = st.columns(2)
                opciones_ejes = ["price", "calificacion_pct", "horas_promedio", "positive_ratings", "propietarios_est"]
                eje_x = c_eje1.selectbox("Eje X:", opciones_ejes, index=0, format_func=lambda x: ETIQUETAS_VARS[x])
                eje_y = c_eje2.selectbox("Eje Y:", opciones_ejes, index=2, format_func=lambda x: ETIQUETAS_VARS[x])

                fig_disp, ax_disp = plt.subplots(figsize=(8, 2.9))
                for seg in [s for s in ORDEN_CLUSTERS if s in segmentos_sel]:
                    sub = df_filtrado[df_filtrado["segmento"] == seg]
                    if not sub.empty:
                        ax_disp.scatter(
                            sub[eje_x],
                            sub[eje_y],
                            c=PALETA_CLUSTERS[seg],
                            label=seg,
                            alpha=0.55,
                            s=25,
                        )
                if eje_x in centroides.columns and eje_y in centroides.columns:
                    cent_vis = centroides[centroides["segmento"].isin(segmentos_sel)]
                    ax_disp.scatter(
                        cent_vis[eje_x],
                        cent_vis[eje_y],
                        c="black",
                        marker="X",
                        s=220,
                        edgecolors="white",
                        linewidth=1.2,
                        label="Centroides",
                        zorder=5,
                    )
                else:
                    cent_extra = df_filtrado.groupby("segmento")[[eje_x, eje_y]].mean()
                    ax_disp.scatter(
                        cent_extra[eje_x],
                        cent_extra[eje_y],
                        c="black",
                        marker="X",
                        s=220,
                        edgecolors="white",
                        linewidth=1.2,
                        label="Promedios de grupo",
                        zorder=5,
                    )
                ax_disp.set_xlabel(ETIQUETAS_VARS[eje_x])
                ax_disp.set_ylabel(ETIQUETAS_VARS[eje_y])
                ax_disp.set_title(f"{ETIQUETAS_VARS[eje_x]} vs {ETIQUETAS_VARS[eje_y]}", fontsize=11, fontweight="bold")
                ax_disp.legend(fontsize=8, loc="best")
                plt.tight_layout()
                st.pyplot(fig_disp)
                plt.close(fig_disp)

        else:
            st.subheader("Comparación de Características entre Clústeres")
            col_c1, col_c2 = st.columns(2)

            orden_vis = [s for s in ORDEN_CLUSTERS if s in df_filtrado["segmento"].unique()]
            etiquetas_cortas = {
                "Económicos de Baja Aceptación": "Baja Acept.",
                "Económicos Bien Valorados": "Bien Valorados",
                "Gama Alta / Premium": "Premium",
                "Éxitos de Alta Retención": "Alta Retención",
            }

            with col_c1:
                fig_comp1, ax_comp1 = plt.subplots(figsize=(7, 2.0))
                sns.boxplot(
                    data=df_filtrado,
                    x="segmento",
                    y="horas_promedio",
                    order=orden_vis,
                    hue="segmento",
                    palette=PALETA_CLUSTERS,
                    legend=False,
                    ax=ax_comp1,
                )
                ax_comp1.set_title("Retención: Horas de Juego Promedio por Clúster", fontsize=11, fontweight="bold")
                ax_comp1.set_xlabel("")
                ax_comp1.set_ylabel("Horas Promedio")
                ax_comp1.set_xticks(range(len(orden_vis)))
                ax_comp1.set_xticklabels([etiquetas_cortas[s] for s in orden_vis], rotation=10)
                plt.tight_layout()
                st.pyplot(fig_comp1)
                plt.close(fig_comp1)

            with col_c2:
                fig_comp2, ax_comp2 = plt.subplots(figsize=(7, 2.0))
                prom_comp = df_filtrado.groupby("segmento")[["price", "calificacion_pct"]].mean().reindex(orden_vis)
                x_idx = range(len(orden_vis))
                width = 0.36
                ax_comp2.bar([i - width / 2 for i in x_idx], prom_comp["price"], width=width, color="#2563eb", label="Precio Promedio (GBP)")
                ax_comp2.bar([i + width / 2 for i in x_idx], prom_comp["calificacion_pct"], width=width, color="#10b981", label="Calificación Positiva (%)")
                ax_comp2.set_xticks(list(x_idx))
                ax_comp2.set_xticklabels([etiquetas_cortas[s] for s in orden_vis], rotation=10)
                ax_comp2.set_title("Comparativa de Precio vs Calificación Promedio por Grupo", fontsize=11, fontweight="bold")
                ax_comp2.set_ylabel("Valor Promedio (GBP / %)")
                ax_comp2.legend(fontsize=8.5)
                plt.tight_layout()
                st.pyplot(fig_comp2)
                plt.close(fig_comp2)

            st.subheader("Estadísticas Descriptivas por Clúster")
            tabla_stats = (
                df_filtrado.groupby("segmento")
                .agg(
                    Juegos=("appid", "count"),
                    Precio_Prom_GBP=("price", "mean"),
                    Precio_DE=("price", "std"),
                    Calif_Prom_Pct=("calificacion_pct", "mean"),
                    Horas_Prom=("horas_promedio", "mean"),
                    Horas_Mediana=("horas_promedio", "median"),
                    Resenas_Pos_Med=("positive_ratings", "median"),
                    Propietarios_Prom=("propietarios_est", "mean"),
                )
                .reindex(orden_vis)
                .round(2)
            )
            st.dataframe(tabla_stats, width="stretch")

            col_d, col_hr = st.columns([1, 1])
            with col_d:
                st.subheader("Descripción de los Grupos Identificados")
                for idx, nombre in enumerate(ORDEN_CLUSTERS):
                    cant = int((df_filtrado["segmento"] == nombre).sum())
                    st.markdown(f"**{idx}. {nombre}** (`{cant:,} juegos`): {DESCRIPCIONES_GRUPOS[nombre]}")

            with col_hr:
                st.subheader("Principales Hallazgos y Recomendaciones")
                st.markdown(
                    "**Hallazgos clave:**\n"
                    "- **El precio económico divide al mercado en dos polos:** *Económicos Bien Valorados* (`4.97 GBP`, `83.41%`) concentra el `52.71%` del catálogo activo, mientras *Económicos de Baja Aceptación* (`3.33 GBP`) cae a `50.90%`.\n"
                    "- **La retención extrema no depende solo del precio máximo:** *Éxitos de Alta Retención* (`16.20 GBP`) promedia `33.52 horas` y `1.49 M` de propietarios, superando casi 5 veces en horas a *Gama Alta / Premium* (`20.58 GBP`, `6.84 horas`).\n"
                    "- **Mayor valoración impulsa más base de jugadores:** los grupos con `~80%` o más superan ampliamente los `400 mil` propietarios estimados frente a `~210 mil` en el grupo de baja aceptación.\n\n"
                    "**Recomendaciones:**\n"
                    "- Evitar competir solo por precio bajo descuidando pulido técnico: en gama `<= 5 GBP`, la calificación define si el título logra ~400 mil o queda estancado.\n"
                    "- Potenciar modos rejugables en lanzamientos de `12 a 20 GBP` para convertir compradores iniciales en comunidades de alta retención (`+30 horas`).\n"
                    "- En juegos *Premium* (`~20 GBP`), ofrecer contenido post-lanzamiento y descuentos estratégicos para prolongar las horas promedio de juego."
                )


if __name__ == "__main__":
    main()
