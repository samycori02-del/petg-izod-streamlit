# ============================================================
# APP STREAMLIT - MODELO PREDICTIVO PETG IZOD + COSTO
# Autor: Proyecto PETG - Resistencia al impacto Izod
# ============================================================

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

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
    page_icon="🧪",
    layout="wide"
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 38px;
        font-weight: 800;
        color: #102A43;
        margin-bottom: 0px;
    }
    .subtitle {
        font-size: 18px;
        color: #52616B;
        margin-bottom: 25px;
    }
    .box {
        background-color: #F8FAFC;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #E2E8F0;
        margin-bottom: 18px;
    }
    .small-note {
        font-size: 14px;
        color: #64748B;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="main-title">Modelo predictivo PETG - Resistencia al impacto Izod</div>
    <div class="subtitle">
    Aplicación para estimar resistencia al impacto Izod en probetas PETG impresas por FDM,
    usando red neuronal MLP, comparación con SVR y estimación de costo productivo.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Configuración de entrada")

archivo = st.sidebar.file_uploader(
    "Sube el archivo Excel de datos",
    type=["xlsx"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Parámetros de costo")

precio_petg_kg = st.sidebar.number_input(
    "Precio PETG (USD/kg)",
    min_value=0.0,
    value=20.00,
    step=0.50
)

masa_por_probeta_g = st.sidebar.number_input(
    "Masa estimada por probeta (g)",
    min_value=0.0,
    value=6.48,
    step=0.01
)

tarifa_kwh = st.sidebar.number_input(
    "Tarifa eléctrica (USD/kWh)",
    min_value=0.0,
    value=0.09,
    step=0.01
)

potencia_kw = st.sidebar.number_input(
    "Potencia promedio impresora (kW)",
    min_value=0.0,
    value=0.25,
    step=0.01
)

costo_maquina_h = st.sidebar.number_input(
    "Costo máquina (USD/h)",
    min_value=0.0,
    value=0.20,
    step=0.05
)

porcentaje_desperdicio = st.sidebar.number_input(
    "Desperdicio (%)",
    min_value=0.0,
    value=5.0,
    step=1.0
) / 100

costo_operador_probeta = st.sidebar.number_input(
    "Costo operador por probeta (USD)",
    min_value=0.0,
    value=0.00,
    step=0.05
)

costo_postproceso = st.sidebar.number_input(
    "Costo postproceso por probeta (USD)",
    min_value=0.0,
    value=0.00,
    step=0.05
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Columnas requeridas en el Excel:\n\n"
    "- Ensayo\n"
    "- Temperatura_c\n"
    "- Altura_capa_m_m\n"
    "- Velocidad_m_m_s\n"
    "- Resistencia_Izod_j_m\n"
    "- Tipo_Dato"
)


# ============================================================
# SI NO HAY ARCHIVO
# ============================================================

if archivo is None:
    st.info("Sube tu archivo Excel desde el panel lateral para iniciar la aplicación.")
    st.stop()


# ============================================================
# CARGA Y LIMPIEZA DE DATOS
# ============================================================

df = pd.read_excel(archivo)

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

df["Tipo_Dato"] = df["Tipo_Dato"].astype(str).str.strip().str.lower()

X = df[["Temperatura_c", "Altura_capa_m_m", "Velocidad_m_m_s"]]
y = df["Resistencia_Izod_j_m"]


# ============================================================
# MODELOS
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

with st.spinner("Evaluando modelo MLP y SVR..."):
    resultados_mlp_cv = cross_validate(
        modelo_mlp,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False
    )

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

comparacion_modelos = pd.DataFrame({
    "Modelo": ["Red neuronal MLP", "SVR"],
    "R2_promedio": [r2_mlp, r2_svr],
    "R2_desviacion": [r2_mlp_std, r2_svr_std],
    "MAE_J_m": [mae_mlp, mae_svr],
    "RMSE_J_m": [rmse_mlp, rmse_svr]
})

modelo_final_mlp = modelo_mlp.fit(X, y)
y_pred_total = modelo_final_mlp.predict(X)

df_predicciones = df.copy()
df_predicciones["Izod_predicho_J_m"] = y_pred_total
df_predicciones["Error"] = (
    df_predicciones["Resistencia_Izod_j_m"]
    - df_predicciones["Izod_predicho_J_m"]
)
df_predicciones["Error_abs"] = np.abs(df_predicciones["Error"])
df_predicciones["Error_porcentual_abs"] = (
    df_predicciones["Error_abs"]
    / df_predicciones["Resistencia_Izod_j_m"]
) * 100


# ============================================================
# COSTOS
# ============================================================

datos_tiempo = [
    ("L1",  1 + 44/60,        5),
    ("L2",  40/60 + 3/3600,   5),
    ("L3",  29/60 + 15/3600,  5),
    ("L4",  1 + 1/60,         5),
    ("L5",  32/60 + 7/3600,   5),
    ("L6",  51/60 + 48/3600,  5),
    ("L7",  42/60 + 1/3600,   5),
    ("L8",  1 + 6/60,         5),
    ("L9",  31/60 + 39/3600,  5),

    ("A1",  13/60 + 1/3600,   1),
    ("A2",  9/60 + 3/3600,    1),
    ("A3",  7/60 + 33/3600,   1),
    ("A4",  6/60 + 34/3600,   1),
    ("A5",  10/60 + 24/3600,  1),
    ("A6",  6/60 + 57/3600,   1),
    ("A7",  11/60 + 8/3600,   1),
    ("A8",  9/60 + 41/3600,   1),
    ("A9",  6/60 + 24/3600,   1),
    ("A10", 12/60 + 55/3600,  1),
    ("A11", 8/60 + 55/3600,   1),
    ("A12", 7/60 + 13/3600,   1),
    ("A13", 10/60 + 2/3600,   1),
    ("A14", 7/60 + 40/3600,   1),
    ("A15", 7/60 + 48/3600,   1),
    ("A16", 10/60 + 2/3600,   1),
    ("A17", 6/60 + 57/3600,   1),
    ("A18", 8/60 + 12/3600,   1),
    ("A19", 11/60 + 8/3600,   1),
    ("A20", 6/60 + 24/3600,   1),
]

df_costos = pd.DataFrame(
    datos_tiempo,
    columns=["Ensayo", "Tiempo_lote_h", "Cantidad_probetas_lote"]
)

df_costos["Tiempo_por_probeta_h"] = (
    df_costos["Tiempo_lote_h"] / df_costos["Cantidad_probetas_lote"]
)

df_costos["Masa_por_probeta_g"] = masa_por_probeta_g

df_costos["Masa_lote_g"] = (
    df_costos["Masa_por_probeta_g"] * df_costos["Cantidad_probetas_lote"]
)

df_costos["Costo_material_USD"] = (
    (df_costos["Masa_por_probeta_g"] / 1000) * precio_petg_kg
)

df_costos["Costo_energia_USD"] = (
    df_costos["Tiempo_por_probeta_h"] * potencia_kw * tarifa_kwh
)

df_costos["Costo_maquina_USD"] = (
    df_costos["Tiempo_por_probeta_h"] * costo_maquina_h
)

df_costos["Costo_desperdicio_USD"] = (
    df_costos["Costo_material_USD"] * porcentaje_desperdicio
)

df_costos["Costo_operador_USD"] = costo_operador_probeta
df_costos["Costo_postproceso_USD"] = costo_postproceso

df_costos["Costo_total_USD"] = (
    df_costos["Costo_material_USD"]
    + df_costos["Costo_energia_USD"]
    + df_costos["Costo_maquina_USD"]
    + df_costos["Costo_operador_USD"]
    + df_costos["Costo_postproceso_USD"]
    + df_costos["Costo_desperdicio_USD"]
)

df_costos["Costo_total_lote_USD"] = (
    df_costos["Costo_total_USD"] * df_costos["Cantidad_probetas_lote"]
)

componentes_costo = pd.DataFrame({
    "Componente": [
        "Material",
        "Energía",
        "Máquina",
        "Desperdicio",
        "Operador",
        "Postproceso"
    ],
    "Costo_promedio_USD": [
        df_costos["Costo_material_USD"].mean(),
        df_costos["Costo_energia_USD"].mean(),
        df_costos["Costo_maquina_USD"].mean(),
        df_costos["Costo_desperdicio_USD"].mean(),
        df_costos["Costo_operador_USD"].mean(),
        df_costos["Costo_postproceso_USD"].mean()
    ]
})


# ============================================================
# TABS
# ============================================================

tab_datos, tab_modelo, tab_prediccion, tab_graficas, tab_costos, tab_exportar = st.tabs([
    "📄 Datos",
    "🧠 Modelo",
    "🎯 Predicción",
    "📊 Gráficas",
    "💵 Costos",
    "⬇️ Exportar"
])


# ============================================================
# TAB 1: DATOS
# ============================================================

with tab_datos:
    st.markdown("### Datos cargados")

    c1, c2, c3 = st.columns(3)
    c1.metric("Número de filas", df.shape[0])
    c2.metric("Número de columnas", df.shape[1])
    c3.metric("Variable respuesta", "Izod J/m")

    st.markdown("#### Vista de datos")
    st.dataframe(df, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Conteo por tipo de dato")
        st.dataframe(df["Tipo_Dato"].value_counts(), use_container_width=True)

    with col2:
        st.markdown("#### Estadísticos descriptivos")
        st.dataframe(
            df[[
                "Temperatura_c",
                "Altura_capa_m_m",
                "Velocidad_m_m_s",
                "Resistencia_Izod_j_m"
            ]].describe(),
            use_container_width=True
        )


# ============================================================
# TAB 2: MODELO
# ============================================================

with tab_modelo:
    st.markdown("### Evaluación del modelo predictivo")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("R² MLP", f"{r2_mlp:.4f}")
    c2.metric("MAE MLP", f"{mae_mlp:.4f} J/m")
    c3.metric("RMSE MLP", f"{rmse_mlp:.4f} J/m")
    c4.metric("Desv. R² MLP", f"{r2_mlp_std:.4f}")

    st.markdown("#### Comparación de modelos")
    st.dataframe(comparacion_modelos, use_container_width=True)

    st.markdown("#### Arquitectura usada")
    st.info(
        "Modelo principal: red neuronal MLP con 3 entradas, "
        "2 capas ocultas de 16 y 8 neuronas, y 1 salida. "
        "Configuración: activation='relu', solver='adam', alpha=0.001."
    )

    fig_comp_modelos, ax_comp_modelos = plt.subplots(figsize=(7, 5))
    ax_comp_modelos.bar(
        comparacion_modelos["Modelo"],
        comparacion_modelos["R2_promedio"]
    )
    ax_comp_modelos.set_ylabel("R² promedio")
    ax_comp_modelos.set_title("Comparación de R² promedio")
    ax_comp_modelos.grid(axis="y", alpha=0.3)
    st.pyplot(fig_comp_modelos)


# ============================================================
# TAB 3: PREDICCIÓN
# ============================================================

with tab_prediccion:
    st.markdown("### Predicción directa")

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

    st.markdown("---")
    st.markdown("### Predicción inversa")

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

    df_busqueda["Diferencia_objetivo"] = np.abs(
        df_busqueda["Izod_predicho_J_m"] - izod_objetivo
    )

    recomendaciones = df_busqueda.sort_values(
        by="Diferencia_objetivo",
        ascending=True
    ).head(10)

    st.markdown("#### Combinaciones recomendadas")
    st.dataframe(recomendaciones, use_container_width=True)

    st.markdown("#### Top 10 combinaciones con mayor Izod predicho")
    mejores_izod = df_busqueda.sort_values(
        by="Izod_predicho_J_m",
        ascending=False
    ).head(10)
    st.dataframe(mejores_izod, use_container_width=True)


# ============================================================
# TAB 4: GRÁFICAS
# ============================================================

with tab_graficas:
    st.markdown("### Gráficas del modelo")

    st.markdown("#### Real vs predicho")

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

    st.markdown("#### Error absoluto por ensayo")

    df_errores = df_predicciones.sort_values(
        by="Error_abs",
        ascending=False
    )

    st.dataframe(
        df_errores[[
            "Ensayo",
            "Temperatura_c",
            "Altura_capa_m_m",
            "Velocidad_m_m_s",
            "Resistencia_Izod_j_m",
            "Izod_predicho_J_m",
            "Error_abs",
            "Tipo_Dato"
        ]].head(15),
        use_container_width=True
    )

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

    st.markdown("#### Distribución del error")

    resumen_errores = pd.DataFrame({
        "Métrica": [
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

    st.dataframe(resumen_errores, use_container_width=True)

    fig3, ax3 = plt.subplots(figsize=(7, 5))
    ax3.hist(df_predicciones["Error"], bins=10)
    ax3.axvline(0, linestyle="--")
    ax3.set_xlabel("Error: real - predicho (J/m)")
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

    st.markdown("#### Importancia de variables")

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

    st.dataframe(df_importancia, use_container_width=True)

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

    st.markdown("#### Mapa de calor")

    altura_fija = st.slider(
        "Altura de capa fija para mapa de calor (mm)",
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
# TAB 5: COSTOS
# ============================================================

with tab_costos:
    st.markdown("### Estimación de costo productivo")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Costo mínimo por probeta",
        f"{df_costos['Costo_total_USD'].min():.4f} USD"
    )

    c2.metric(
        "Costo promedio por probeta",
        f"{df_costos['Costo_total_USD'].mean():.4f} USD"
    )

    c3.metric(
        "Costo máximo por probeta",
        f"{df_costos['Costo_total_USD'].max():.4f} USD"
    )

    st.markdown("#### Tabla de costos por probeta")

    st.dataframe(
        df_costos[[
            "Ensayo",
            "Cantidad_probetas_lote",
            "Tiempo_lote_h",
            "Tiempo_por_probeta_h",
            "Masa_por_probeta_g",
            "Masa_lote_g",
            "Costo_material_USD",
            "Costo_energia_USD",
            "Costo_maquina_USD",
            "Costo_desperdicio_USD",
            "Costo_operador_USD",
            "Costo_postproceso_USD",
            "Costo_total_USD",
            "Costo_total_lote_USD"
        ]],
        use_container_width=True
    )

    st.markdown("#### Costo total por probeta")

    fig_costo, ax_costo = plt.subplots(figsize=(12, 5))

    ax_costo.bar(
        df_costos["Ensayo"],
        df_costos["Costo_total_USD"]
    )

    ax_costo.set_xlabel("Ensayo")
    ax_costo.set_ylabel("Costo total por probeta (USD)")
    ax_costo.set_title("Costo productivo estimado por probeta")
    ax_costo.tick_params(axis="x", rotation=90)
    ax_costo.grid(axis="y", alpha=0.3)
    st.pyplot(fig_costo)

    st.markdown("#### Desglose promedio del costo")

    st.dataframe(componentes_costo, use_container_width=True)

    fig_comp, ax_comp = plt.subplots(figsize=(7, 5))

    ax_comp.bar(
        componentes_costo["Componente"],
        componentes_costo["Costo_promedio_USD"]
    )

    ax_comp.set_xlabel("Componente")
    ax_comp.set_ylabel("Costo promedio (USD/probeta)")
    ax_comp.set_title("Desglose promedio del costo productivo")
    ax_comp.tick_params(axis="x", rotation=45)
    ax_comp.grid(axis="y", alpha=0.3)
    st.pyplot(fig_comp)


# ============================================================
# TAB 6: EXPORTAR
# ============================================================

with tab_exportar:
    st.markdown("### Exportar resultados")

    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Datos_originales", index=False)
        comparacion_modelos.to_excel(writer, sheet_name="Comparacion_modelos", index=False)
        df_predicciones.to_excel(writer, sheet_name="Predicciones_MLP", index=False)
        df_importancia.to_excel(writer, sheet_name="Importancia_variables", index=False)
        recomendaciones.to_excel(writer, sheet_name="Prediccion_inversa", index=False)
        mejores_izod.to_excel(writer, sheet_name="Mejores_Izod", index=False)
        df_costos.to_excel(writer, sheet_name="Costos", index=False)
        componentes_costo.to_excel(writer, sheet_name="Componentes_costo", index=False)

    st.download_button(
        label="Descargar resultados en Excel",
        data=output.getvalue(),
        file_name="resultados_petg_izod.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.markdown("#### Resumen técnico")
    st.write(
        f"""
        Modelo principal: Red neuronal MLP  
        Arquitectura: 3 entradas → 16 neuronas → 8 neuronas → 1 salida  
        R² promedio MLP: {r2_mlp:.4f}  
        MAE MLP: {mae_mlp:.4f} J/m  
        RMSE MLP: {rmse_mlp:.4f} J/m  
        Modelo comparativo: SVR  
        R² promedio SVR: {r2_svr:.4f}
        """
    )
