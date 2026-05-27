import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.inspection import permutation_importance


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="PETG Izod - Modelo Predictivo",
    layout="wide"
)

st.title("Modelo predictivo PETG - Resistencia al impacto Izod")
st.write(
    "Aplicación en Streamlit para predecir la resistencia al impacto Izod "
    "en probetas PETG fabricadas mediante impresión 3D FDM."
)


# ============================================================
# CARGA DE DATOS
# ============================================================

st.sidebar.header("Carga de datos")

archivo = st.sidebar.file_uploader(
    "Sube el archivo Excel",
    type=["xlsx"]
)

st.sidebar.markdown(
    """
    **Columnas requeridas:**
    - Ensayo
    - Temperatura_c
    - Altura_capa_m_m
    - Velocidad_m_m_s
    - Resistencia_Izod_j_m
    - Tipo_Dato
    """
)

if archivo is None:
    st.info("Sube tu archivo Excel para iniciar la aplicación.")
    st.stop()


df = pd.read_excel(archivo)

# Limpiar nombres de columnas por si tienen espacios
df.columns = df.columns.astype(str).str.strip()

columnas_necesarias = [
    "Ensayo",
    "Temperatura_c",
    "Altura_capa_m_m",
    "Velocidad_m_m_s",
    "Resistencia_Izod_j_m",
    "Tipo_Dato"
]

faltantes = [col for col in columnas_necesarias if col not in df.columns]

if faltantes:
    st.error(f"Faltan columnas en el Excel: {faltantes}")
    st.stop()

# Limpieza de Tipo_Dato
df["Tipo_Dato"] = df["Tipo_Dato"].astype(str).str.strip().str.lower()

st.subheader("1. Datos cargados")
st.write(f"Filas: {df.shape[0]} | Columnas: {df.shape[1]}")
st.dataframe(df)

col_a, col_b = st.columns(2)

with col_a:
    st.write("Conteo por tipo de dato")
    st.dataframe(df["Tipo_Dato"].value_counts())

with col_b:
    st.write("Estadísticos descriptivos")
    st.dataframe(df[[
        "Temperatura_c",
        "Altura_capa_m_m",
        "Velocidad_m_m_s",
        "Resistencia_Izod_j_m"
    ]].describe())


# ============================================================
# VARIABLES DEL MODELO
# ============================================================

X = df[["Temperatura_c", "Altura_capa_m_m", "Velocidad_m_m_s"]]
y = df["Resistencia_Izod_j_m"]


# ============================================================
# MODELO MLP - RED NEURONAL
# ============================================================

modelo_mlp = TransformedTargetRegressor(
    regressor=Pipeline([
        ("scaler_X", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(16, 8),
            activation="relu",
            solver="adam",
            alpha=0.001,
            max_iter=10000,
            random_state=42
        ))
    ]),
    transformer=StandardScaler()
)

modelo_svr = TransformedTargetRegressor(
    regressor=Pipeline([
        ("scaler_X", StandardScaler()),
        ("svr", SVR(
            kernel="rbf",
            C=1.0,
            epsilon=0.1,
            gamma="scale"
        ))
    ]),
    transformer=StandardScaler()
)


# ============================================================
# VALIDACIÓN CRUZADA
# ============================================================

st.subheader("2. Evaluación del modelo")

cv = RepeatedKFold(
    n_splits=5,
    n_repeats=10,
    random_state=42
)

scoring = {
    "R2": "r2",
    "MAE": "neg_mean_absolute_error",
    "RMSE": "neg_root_mean_squared_error"
}

with st.spinner("Evaluando red neuronal MLP con validación cruzada..."):
    resultados_mlp_cv = cross_validate(
        modelo_mlp,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False
    )

with st.spinner("Evaluando modelo SVR comparativo..."):
    resultados_svr_cv = cross_validate(
        modelo_svr,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False
    )

r2_mlp = np.mean(resultados_mlp_cv["test_R2"])
r2_mlp_std = np.std(resultados_mlp_cv["test_R2"])
mae_mlp = -np.mean(resultados_mlp_cv["test_MAE"])
rmse_mlp = -np.mean(resultados_mlp_cv["test_RMSE"])

r2_svr = np.mean(resultados_svr_cv["test_R2"])
r2_svr_std = np.std(resultados_svr_cv["test_R2"])
mae_svr = -np.mean(resultados_svr_cv["test_MAE"])
rmse_svr = -np.mean(resultados_svr_cv["test_RMSE"])

col1, col2, col3, col4 = st.columns(4)

col1.metric("R² MLP", f"{r2_mlp:.4f}")
col2.metric("MAE MLP", f"{mae_mlp:.4f} J/m")
col3.metric("RMSE MLP", f"{rmse_mlp:.4f} J/m")
col4.metric("Desv. R² MLP", f"{r2_mlp_std:.4f}")

comparacion_modelos = pd.DataFrame({
    "Modelo": ["Red neuronal MLP", "SVR"],
    "R2_promedio": [r2_mlp, r2_svr],
    "R2_desviacion": [r2_mlp_std, r2_svr_std],
    "MAE_J_m": [mae_mlp, mae_svr],
    "RMSE_J_m": [rmse_mlp, rmse_svr]
})

st.write("Comparación MLP vs SVR")
st.dataframe(comparacion_modelos)


# ============================================================
# ENTRENAMIENTO FINAL CON TODOS LOS DATOS
# ============================================================

modelo_final_mlp = modelo_mlp.fit(X, y)

st.success(
    "Modelo final MLP entrenado con todos los datos disponibles. "
    "Arquitectura: 3 entradas → 16 neuronas → 8 neuronas → 1 salida."
)


# ============================================================
# PREDICCIÓN DIRECTA
# ============================================================

st.subheader("3. Predicción directa de resistencia Izod")

col1, col2, col3 = st.columns(3)

with col1:
    temperatura = st.slider(
        "Temperatura de impresión (°C)",
        min_value=230,
        max_value=260,
        value=245,
        step=1
    )

with col2:
    altura = st.slider(
        "Altura de capa (mm)",
        min_value=0.12,
        max_value=0.28,
        value=0.20,
        step=0.01
    )

with col3:
    velocidad = st.slider(
        "Velocidad de impresión (mm/s)",
        min_value=60,
        max_value=300,
        value=120,
        step=10
    )

nuevo_dato = pd.DataFrame({
    "Temperatura_c": [temperatura],
    "Altura_capa_m_m": [altura],
    "Velocidad_m_m_s": [velocidad]
})

prediccion = modelo_final_mlp.predict(nuevo_dato)[0]

st.success(f"Resistencia Izod predicha: {prediccion:.4f} J/m")


# ============================================================
# REAL VS PREDICHO
# ============================================================

st.subheader("4. Comparación real vs predicho")

y_pred_total = modelo_final_mlp.predict(X)

df_predicciones = df.copy()
df_predicciones["Izod_predicho_J_m"] = y_pred_total
df_predicciones["Error"] = (
    df_predicciones["Resistencia_Izod_j_m"]
    - df_predicciones["Izod_predicho_J_m"]
)
df_predicciones["Error_abs"] = abs(df_predicciones["Error"])
df_predicciones["Error_porcentual_abs"] = (
    df_predicciones["Error_abs"]
    / df_predicciones["Resistencia_Izod_j_m"]
) * 100

st.dataframe(df_predicciones)

fig1, ax1 = plt.subplots(figsize=(7, 6))

ax1.scatter(
    df_predicciones["Resistencia_Izod_j_m"],
    df_predicciones["Izod_predicho_J_m"]
)

min_val = min(
    df_predicciones["Resistencia_Izod_j_m"].min(),
    df_predicciones["Izod_predicho_J_m"].min()
)

max_val = max(
    df_predicciones["Resistencia_Izod_j_m"].max(),
    df_predicciones["Izod_predicho_J_m"].max()
)

ax1.plot([min_val, max_val], [min_val, max_val])
ax1.set_xlabel("Resistencia Izod real (J/m)")
ax1.set_ylabel("Resistencia Izod predicha (J/m)")
ax1.set_title("Comparación entre resistencia Izod real y predicha")
ax1.grid(True)

st.pyplot(fig1)


# ============================================================
# ERROR ABSOLUTO POR ENSAYO
# ============================================================

st.subheader("5. Error absoluto por ensayo")

df_errores = df_predicciones.sort_values(
    by="Error_abs",
    ascending=False
)

st.write("Ensayos con mayor error")
st.dataframe(df_errores[[
    "Ensayo",
    "Temperatura_c",
    "Altura_capa_m_m",
    "Velocidad_m_m_s",
    "Resistencia_Izod_j_m",
    "Izod_predicho_J_m",
    "Error_abs",
    "Tipo_Dato"
]].head(15))

fig2, ax2 = plt.subplots(figsize=(12, 5))

ax2.bar(
    df_errores["Ensayo"].astype(str),
    df_errores["Error_abs"]
)

ax2.set_xlabel("Ensayo")
ax2.set_ylabel("Error absoluto (J/m)")
ax2.set_title("Error absoluto de predicción por ensayo")
ax2.tick_params(axis="x", rotation=90)
ax2.grid(axis="y", alpha=0.3)

st.pyplot(fig2)


# ============================================================
# DISTRIBUCIÓN DEL ERROR
# ============================================================

st.subheader("6. Distribución del error")

resumen_errores = pd.DataFrame({
    "Metrica": [
        "Error medio",
        "Error absoluto medio",
        "Error máximo absoluto",
        "Error porcentual absoluto medio"
    ],
    "Valor": [
        df_predicciones["Error"].mean(),
        df_predicciones["Error_abs"].mean(),
        df_predicciones["Error_abs"].max(),
        df_predicciones["Error_porcentual_abs"].mean()
    ]
})

st.dataframe(resumen_errores)

fig3, ax3 = plt.subplots(figsize=(7, 5))

ax3.hist(df_predicciones["Error"], bins=10)
ax3.axvline(0, linestyle="--")
ax3.set_xlabel("Error de predicción: Real - Predicho (J/m)")
ax3.set_ylabel("Frecuencia")
ax3.set_title("Distribución del error de predicción")
ax3.grid(True, alpha=0.3)

st.pyplot(fig3)

fig4, ax4 = plt.subplots(figsize=(7, 5))

ax4.hist(df_predicciones["Error_abs"], bins=10)
ax4.set_xlabel("Error absoluto (J/m)")
ax4.set_ylabel("Frecuencia")
ax4.set_title("Distribución del error absoluto")
ax4.grid(True, alpha=0.3)

st.pyplot(fig4)


# ============================================================
# IMPORTANCIA DE VARIABLES
# ============================================================

st.subheader("7. Importancia de variables")

with st.spinner("Calculando importancia de variables..."):
    importancia = permutation_importance(
        modelo_final_mlp,
        X,
        y,
        n_repeats=30,
        random_state=42,
        scoring="r2"
    )

df_importancia = pd.DataFrame({
    "Variable": X.columns,
    "Importancia_media": importancia.importances_mean,
    "Importancia_desviacion": importancia.importances_std
}).sort_values(
    by="Importancia_media",
    ascending=False
)

st.dataframe(df_importancia)

fig5, ax5 = plt.subplots(figsize=(7, 5))

ax5.bar(
    df_importancia["Variable"],
    df_importancia["Importancia_media"]
)

ax5.set_ylabel("Importancia media")
ax5.set_xlabel("Variable")
ax5.set_title("Importancia de variables en el modelo MLP")
ax5.grid(axis="y", alpha=0.3)

st.pyplot(fig5)


# ============================================================
# MAPA DE CALOR
# ============================================================

st.subheader("8. Mapa de calor de resistencia Izod predicha")

altura_fija = st.slider(
    "Altura de capa fija para el mapa de calor (mm)",
    min_value=0.12,
    max_value=0.28,
    value=0.20,
    step=0.01
)

temperaturas = np.linspace(230, 260, 80)
velocidades = np.linspace(60, 300, 80)

T_grid, V_grid = np.meshgrid(temperaturas, velocidades)

datos_malla = pd.DataFrame({
    "Temperatura_c": T_grid.ravel(),
    "Altura_capa_m_m": altura_fija,
    "Velocidad_m_m_s": V_grid.ravel()
})

Z_pred = modelo_final_mlp.predict(datos_malla)
Z_grid = Z_pred.reshape(T_grid.shape)

fig6, ax6 = plt.subplots(figsize=(9, 6))

contorno = ax6.contourf(
    T_grid,
    V_grid,
    Z_grid,
    levels=20,
    cmap="viridis"
)

fig6.colorbar(
    contorno,
    ax=ax6,
    label="Resistencia Izod predicha (J/m)"
)

ax6.set_xlabel("Temperatura de impresión (°C)")
ax6.set_ylabel("Velocidad de impresión (mm/s)")
ax6.set_title(
    f"Mapa de calor de resistencia Izod predicha\nAltura fija = {altura_fija:.2f} mm"
)
ax6.grid(alpha=0.3)

st.pyplot(fig6)


# ============================================================
# PREDICCIÓN INVERSA
# ============================================================

st.subheader("9. Predicción inversa")

izod_objetivo = st.slider(
    "Resistencia Izod objetivo (J/m)",
    min_value=14.0,
    max_value=35.0,
    value=25.0,
    step=0.1
)

temperaturas_inv = np.arange(230, 261, 1)
alturas_inv = np.arange(0.12, 0.281, 0.01)
velocidades_inv = np.arange(60, 301, 10)

combinaciones = []

for temp in temperaturas_inv:
    for alt in alturas_inv:
        for vel in velocidades_inv:
            combinaciones.append([temp, round(alt, 2), vel])

df_busqueda = pd.DataFrame(
    combinaciones,
    columns=[
        "Temperatura_c",
        "Altura_capa_m_m",
        "Velocidad_m_m_s"
    ]
)

df_busqueda["Izod_predicho_J_m"] = modelo_final_mlp.predict(df_busqueda)

df_busqueda["Diferencia_objetivo"] = abs(
    df_busqueda["Izod_predicho_J_m"] - izod_objetivo
)

recomendaciones = df_busqueda.sort_values(
    by="Diferencia_objetivo",
    ascending=True
).head(10)

st.write(f"Combinaciones más cercanas a {izod_objetivo:.2f} J/m")
st.dataframe(recomendaciones)


# ============================================================
# MEJORES COMBINACIONES PARA MAXIMIZAR IZOD
# ============================================================

st.subheader("10. Combinaciones con mayor resistencia Izod predicha")

mejores_izod = df_busqueda.sort_values(
    by="Izod_predicho_J_m",
    ascending=False
).head(10)

st.dataframe(mejores_izod)


# ============================================================
# ANÁLISIS EXPLORATORIO DE REPETICIONES
# ============================================================

st.subheader("11. Análisis exploratorio de repeticiones")

df_repeticion = df[df["Tipo_Dato"] == "repeticion"].copy()

if len(df_repeticion) > 0:
    df_repeticion["Grupo"] = (
        df_repeticion["Ensayo"]
        .astype(str)
        .str.split("-")
        .str[0]
    )

    df_repeticion_promedio = df_repeticion.groupby("Grupo").agg({
        "Temperatura_c": "first",
        "Altura_capa_m_m": "first",
        "Velocidad_m_m_s": "first",
        "Resistencia_Izod_j_m": ["mean", "std", "count"]
    }).reset_index()

    df_repeticion_promedio.columns = [
        "Grupo",
        "Temperatura_c",
        "Altura_capa_m_m",
        "Velocidad_m_m_s",
        "Izod_promedio_J_m",
        "Izod_desviacion_J_m",
        "Numero_probetas"
    ]

    st.dataframe(df_repeticion_promedio)

else:
    st.info("No se encontraron datos con Tipo_Dato = repeticion.")


# ============================================================
# DESCARGA DE RESULTADOS
# ============================================================

st.subheader("12. Descargar resultados")

archivo_salida = "resultados_modelo_izod.xlsx"

with pd.ExcelWriter(archivo_salida, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="Datos_originales", index=False)
    comparacion_modelos.to_excel(writer, sheet_name="Comparacion_modelos", index=False)
    df_predicciones.to_excel(writer, sheet_name="Predicciones_MLP", index=False)
    df_importancia.to_excel(writer, sheet_name="Importancia_variables", index=False)
    recomendaciones.to_excel(writer, sheet_name="Prediccion_inversa", index=False)
    mejores_izod.to_excel(writer, sheet_name="Mejores_combinaciones", index=False)

with open(archivo_salida, "rb") as f:
    st.download_button(
        label="Descargar resultados en Excel",
        data=f,
        file_name="resultados_modelo_izod.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
