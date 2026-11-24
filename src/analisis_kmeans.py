import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

pd.set_option("display.max_columns", 12)
pd.set_option("display.width", 140)

# 1. Cargar y explorar el dataset real de Steam (Kaggle)
df = pd.read_csv("datos/steam.csv")
total_inicial = len(df)

print("=== 1. EXPLORACION INICIAL DEL DATASET (steam.csv) ===")
print("Dimensiones:", df.shape[0], "filas x", df.shape[1], "columnas")
print("Columnas:", list(df.columns))
print("\nPrimeros 10 registros:")
print(df[["appid", "name", "release_date", "genres", "positive_ratings", "negative_ratings", "average_playtime", "price"]].head(10))
print("\nTipos de datos:")
print(df.dtypes)
print("\nEstadisticas descriptivas:")
print(df.describe().round(2))
print("\nValores nulos por columna:")
print(df.isnull().sum())
print("\nRegistros duplicados exactos:", df.duplicated().sum())

# Crear columnas derivadas temporales para explorar el catalogo original
df["anio_lanzamiento"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
df["genero_principal"] = df["genres"].astype(str).str.split(";").str[0]
df["total_resenas"] = df["positive_ratings"] + df["negative_ratings"]
df["calificacion_pct"] = (df["positive_ratings"] / df["total_resenas"].replace(0, pd.NA)) * 100
df["calificacion_pct"] = pd.to_numeric(df["calificacion_pct"], errors="coerce")
df["horas_promedio"] = df["average_playtime"] / 60.0

# Grafica exploratoria 1: Distribucion de variables numericas (con outliers reales)
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
sns.histplot(df["price"], bins=30, color="#3b82f6")
plt.title("Precio en Steam (GBP)")

plt.subplot(2, 2, 2)
sns.histplot(df["calificacion_pct"].dropna(), bins=25, color="#10b981")
plt.title("Porcentaje de resenas positivas (%)")

plt.subplot(2, 2, 3)
sns.histplot(df[df["average_playtime"] > 0]["horas_promedio"], bins=30, color="#8b5cf6")
plt.title("Horas de juego promedio (> 0, con extremos)")

plt.subplot(2, 2, 4)
sns.histplot(df[df["positive_ratings"] < 50000]["positive_ratings"], bins=30, color="#ef4444")
plt.title("Resenas positivas (< 50,000)")

plt.tight_layout()
plt.show()

# Grafica exploratoria 2: Mapa de calor de correlaciones
cols_corr = ["anio_lanzamiento", "price", "calificacion_pct", "positive_ratings", "horas_promedio", "achievements"]
plt.figure(figsize=(9, 6))
sns.heatmap(df[cols_corr].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Mapa de calor de correlaciones (Steam real)")
plt.tight_layout()
plt.show()

# Grafica exploratoria 3: Top 10 generos principales y su precio
top_generos = df["genero_principal"].value_counts().head(10).index
df_top_gen = df[df["genero_principal"].isin(top_generos)]

plt.figure(figsize=(13, 5))
plt.subplot(1, 2, 1)
sns.countplot(data=df_top_gen, y="genero_principal", order=top_generos, color="#3b82f6")
plt.title("Top 10 generos en el catalogo")
plt.xlabel("Cantidad de juegos")
plt.ylabel("Genero principal")

plt.subplot(1, 2, 2)
sns.boxplot(data=df_top_gen[df_top_gen["price"] <= 60], x="price", y="genero_principal", order=top_generos, color="#10b981")
plt.title("Precio por genero (<= 60 GBP)")
plt.xlabel("Precio (GBP)")
plt.ylabel("")

plt.tight_layout()
plt.show()

# 2. Limpieza y preparacion de datos
# Rellenar valores nulos en developer y publisher
nulos_antes = int(df[["developer", "publisher"]].isnull().sum().sum())
df["developer"] = df["developer"].fillna("Desconocido")
df["publisher"] = df["publisher"].fillna("Desconocido")

# Eliminar filas sin fecha o calificacion valida y eliminar duplicados exactos
df = df.dropna(subset=["anio_lanzamiento", "calificacion_pct"]).copy()
df["anio_lanzamiento"] = df["anio_lanzamiento"].astype(int)
antes_dup = len(df)
df = df.drop_duplicates().reset_index(drop=True)
duplicados_quitados = antes_dup - len(df)

# Filtrar juegos sin tiempo registrado, con muy pocas resenas y valores atipicos/software caro
sin_horas_o_pocas_resenas = int(((df["average_playtime"] <= 0) | (df["total_resenas"] < 20)).sum())
df = df[(df["average_playtime"] > 0) & (df["total_resenas"] >= 20)].copy()

antes_atipicos = len(df)
df = df[(df["price"] >= 0) & (df["price"] <= 60) & (df["horas_promedio"] <= 60)].reset_index(drop=True)
atipicos_quitados = antes_atipicos - len(df)

df["calificacion_pct"] = df["calificacion_pct"].round(2)
df["horas_promedio"] = df["horas_promedio"].round(2)

# Calcular propietarios estimados promedio a partir del rango 'min-max' de owners
partes_owners = df["owners"].astype(str).str.split("-", expand=True)
df["propietarios_est"] = ((partes_owners[0].astype(float) + partes_owners[1].astype(float)) / 2).astype(int)

# Codificar variables categoricas con LabelEncoder
le = LabelEncoder()
df["genero_cod"] = le.fit_transform(df["genero_principal"])
df["plataforma_cod"] = le.fit_transform(df["platforms"])

# Seleccionar y escalar las 3 variables para K-Means con StandardScaler
variables = ["price", "calificacion_pct", "horas_promedio"]
scaler = StandardScaler()
X = scaler.fit_transform(df[variables])
df_escalado = pd.DataFrame(X, columns=variables)

print("\n=== 2. RESULTADOS DE LIMPIEZA Y ESCALAMIENTO ===")
print("Registros iniciales (steam.csv):", total_inicial)
print("Nulos imputados en developer/publisher:", nulos_antes)
print("Duplicados exactos eliminados:", duplicados_quitados)
print("Descartados sin playtime > 0 o < 20 resenas:", sin_horas_o_pocas_resenas)
print("Atipicos eliminados (precio > 60 o horas > 60):", atipicos_quitados)
print("Registros limpios finales para clustering:", len(df))
print("Nulos finales:", int(df.isnull().sum().sum()))
print("\nPrimeras 5 filas escaladas con StandardScaler:")
print(df_escalado.head(5).round(3))

# Grafica comparativa antes y despues de limpiar y escalar
plt.figure(figsize=(13, 5))
plt.subplot(1, 2, 1)
sns.boxplot(data=df[["price", "calificacion_pct", "horas_promedio"]], orient="h")
plt.title("Despues de limpiar (escala original)")

plt.subplot(1, 2, 2)
sns.boxplot(data=df_escalado, orient="h")
plt.axvline(0, color="black", linestyle="--")
plt.title("Despues de aplicar StandardScaler")

plt.tight_layout()
plt.show()

# 3. Evaluar K desde 2 hasta 7
valores_k = [2, 3, 4, 5, 6, 7]
resultados_k = []

for k in valores_k:
    modelo_k = KMeans(n_clusters=k, random_state=42, n_init=10)
    grupos_k = modelo_k.fit_predict(X)
    resultados_k.append({
        "K": k,
        "Inercia_WCSS": round(float(modelo_k.inertia_), 2),
        "Silhouette_Score": round(float(silhouette_score(X, grupos_k)), 4),
        "Iteraciones": int(modelo_k.n_iter_),
    })

df_eval = pd.DataFrame(resultados_k)
print("\n=== 3. EVALUACION DE VALORES DE K ===")
print(df_eval.to_string(index=False))

# Grafica del Metodo del Codo y Coeficiente de Silueta
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(df_eval["K"], df_eval["Inercia_WCSS"], marker="o", color="#2563eb")
plt.axvline(4, color="red", linestyle="--", label="K seleccionado = 4")
plt.title("Metodo del Codo (Inercia vs K)")
plt.xlabel("Numero de grupos (K)")
plt.ylabel("Inercia")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(df_eval["K"], df_eval["Silhouette_Score"], marker="s", color="#059669")
plt.axvline(4, color="red", linestyle="--", label="K = 4 (mejor equilibrio)")
plt.title("Coeficiente de Silueta vs K")
plt.xlabel("Numero de grupos (K)")
plt.ylabel("Silhouette Score")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 4. Entrenar modelo final con K = 4
modelo = KMeans(n_clusters=4, random_state=42, n_init=10)
df["Grupo"] = modelo.fit_predict(X)

# Ordenar grupos por horas promedio y asignar nombres descriptivos (estilo p3.py)
promedios = df.groupby("Grupo")["horas_promedio"].mean().sort_values()
nombres = [
    "Económicos de Baja Aceptación",
    "Económicos Bien Valorados",
    "Gama Alta / Premium",
    "Éxitos de Alta Retención",
]
mapa_nombres = dict(zip(promedios.index, nombres))
mapa_ids = dict(zip(nombres, [0, 1, 2, 3]))

df["segmento"] = df["Grupo"].map(mapa_nombres)
df["cluster"] = df["segmento"].map(mapa_ids)
df = df.drop(columns=["Grupo"])

centroides = pd.DataFrame(scaler.inverse_transform(modelo.cluster_centers_), columns=variables).round(2)
centroides["segmento"] = centroides.index.map(mapa_nombres)
centroides["cluster"] = centroides["segmento"].map(mapa_ids)
centroides = centroides.sort_values("cluster").reset_index(drop=True)

conteo = df["segmento"].value_counts().reindex(nombres)
porcentajes = (conteo / len(df) * 100).round(2)
medias_grupo = df.groupby("segmento")[variables + ["positive_ratings", "propietarios_est"]].mean().reindex(nombres).round(2)

inercia_k4 = round(float(modelo.inertia_), 2)
sil_k4 = round(float(silhouette_score(X, df["cluster"])), 4)

print("\n=== 4. RESULTADOS DEL MODELO K-MEANS (K = 4) ===")
print("Inercia:", inercia_k4, "| Silhouette:", sil_k4, "| Iteraciones:", modelo.n_iter_)
print("\nCentroides en escala original:")
print(centroides.to_string(index=False))
print("\nCantidad de juegos por grupo:")
print(conteo)
print("\nPromedios por grupo:")
print(medias_grupo)

colores = {
    "Económicos de Baja Aceptación": "#ef4444",
    "Económicos Bien Valorados": "#3b82f6",
    "Gama Alta / Premium": "#f59e0b",
    "Éxitos de Alta Retención": "#10b981",
}

# Grafica de agrupamientos con K = 4 y centroides
plt.figure(figsize=(13, 5.5))

plt.subplot(1, 2, 1)
for nombre in nombres:
    parte = df[df["segmento"] == nombre]
    plt.scatter(parte["price"], parte["calificacion_pct"], color=colores[nombre], alpha=0.55, s=25, label=nombre)
plt.scatter(centroides["price"], centroides["calificacion_pct"], color="black", marker="X", s=220, edgecolors="white", label="Centroides", zorder=3)
plt.title("Precio (GBP) vs Calificacion positiva (K = 4)")
plt.xlabel("Precio (GBP)")
plt.ylabel("Calificacion positiva (%)")
plt.legend(fontsize=8)
plt.grid(True)

plt.subplot(1, 2, 2)
for nombre in nombres:
    parte = df[df["segmento"] == nombre]
    plt.scatter(parte["price"], parte["horas_promedio"], color=colores[nombre], alpha=0.55, s=25, label=nombre)
plt.scatter(centroides["price"], centroides["horas_promedio"], color="black", marker="X", s=220, edgecolors="white", label="Centroides", zorder=3)
plt.title("Precio (GBP) vs Horas de juego promedio (K = 4)")
plt.xlabel("Precio (GBP)")
plt.ylabel("Horas promedio")
plt.legend(fontsize=8)
plt.grid(True)

plt.tight_layout()
plt.show()

# 5. Comparar configuraciones: K = 3 vs K = 4
modelo_k3 = KMeans(n_clusters=3, random_state=42, n_init=10)
grupos_k3 = modelo_k3.fit_predict(X)
centroides_k3 = pd.DataFrame(scaler.inverse_transform(modelo_k3.cluster_centers_), columns=variables)
inercia_k3 = round(float(modelo_k3.inertia_), 2)
sil_k3 = round(float(silhouette_score(X, grupos_k3)), 4)

print("\n=== 5. COMPARACION ENTRE K = 3 Y K = 4 ===")
print("K = 3 -> Inercia:", inercia_k3, "| Silhouette:", sil_k3)
print("K = 4 -> Inercia:", inercia_k4, "| Silhouette:", sil_k4)

plt.figure(figsize=(13, 5.5))

plt.subplot(1, 2, 1)
plt.scatter(df["price"], df["horas_promedio"], c=grupos_k3, cmap="viridis", alpha=0.55, s=25)
plt.scatter(centroides_k3["price"], centroides_k3["horas_promedio"], color="red", marker="X", s=220, edgecolors="black", label="Centroides K=3", zorder=3)
plt.title(f"K = 3 (Silhouette = {sil_k3})")
plt.xlabel("Precio (GBP)")
plt.ylabel("Horas promedio")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
for nombre in nombres:
    parte = df[df["segmento"] == nombre]
    plt.scatter(parte["price"], parte["horas_promedio"], color=colores[nombre], alpha=0.55, s=25, label=nombre)
plt.scatter(centroides["price"], centroides["horas_promedio"], color="black", marker="X", s=220, edgecolors="white", label="Centroides K=4", zorder=3)
plt.title(f"K = 4 (Silhouette = {sil_k4})")
plt.xlabel("Precio (GBP)")
plt.ylabel("Horas promedio")
plt.legend(fontsize=8)
plt.grid(True)

plt.tight_layout()
plt.show()

# Grafica comparativa de perfiles y cantidad por segmento
centroides_std = pd.DataFrame(modelo.cluster_centers_, columns=variables)
centroides_std["segmento"] = centroides_std.index.map(mapa_nombres)
centroides_std = centroides_std.set_index("segmento").reindex(nombres)

plt.figure(figsize=(13, 5.5))

plt.subplot(1, 2, 1)
centroides_std.T.plot(kind="bar", ax=plt.gca(), color=[colores[n] for n in nombres])
plt.axhline(0, color="black", linestyle="--")
plt.title("Perfil estandarizado de centroides (Puntuacion Z)")
plt.ylabel("Desviaciones estandar")
plt.xticks(rotation=15, ha="right")
plt.legend(fontsize=8)

plt.subplot(1, 2, 2)
plt.bar(["Baja acept.", "Bien valorados", "Premium", "Alta retencion"], conteo.values, color=[colores[n] for n in nombres])
plt.title("Cantidad de videojuegos por grupo")
plt.ylabel("Cantidad")
plt.grid(axis="y")

plt.tight_layout()
plt.show()

# 6. Guardar dataset procesado y metricas para el dashboard
df.to_csv("datos/steam_procesado.csv", index=False)
centroides.to_csv("datos/centroides_kmeans.csv", index=False)
df_eval.to_csv("datos/metricas_evaluacion_k.csv", index=False)

resumen = {
    "registros_iniciales": total_inicial,
    "nulos_imputados": nulos_antes,
    "duplicados_eliminados": duplicados_quitados,
    "sin_horas_o_resenas": sin_horas_o_pocas_resenas,
    "atipicos_eliminados": atipicos_quitados,
    "registros_limpios": len(df),
    "k_optimo": 4,
    "inercia_k4": inercia_k4,
    "silhouette_k4": sil_k4,
    "iteraciones_k4": int(modelo.n_iter_),
    "inercia_k3": inercia_k3,
    "silhouette_k3": sil_k3,
    "conteo_clusters": {k: int(v) for k, v in conteo.items()},
    "porcentaje_clusters": {k: float(v) for k, v in porcentajes.items()},
    "medias_grupo": medias_grupo.to_dict(orient="index"),
}
with open("datos/resumen_modelo.json", "w", encoding="utf-8") as f:
    json.dump(resumen, f, indent=2, ensure_ascii=False)

print("\nResultados y dataset procesado guardados en datos/ correctamente.")
