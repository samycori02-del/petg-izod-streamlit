# ============================================================
# APP STREAMLIT - PETG FDM
# Modelo predictivo de resistencia al impacto + costo productivo
# Versión para exposición interactiva
# ============================================================

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="PETG FDM - Resistencia al impacto",
    page_icon="🧪",
    layout="wide"
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 40px;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0px;
    }
    .subtitle {
        font-size: 18px;
        color: #475569;
        margin-bottom: 20px;
    }
    .info-card {
        background-color: #F8FAFC;
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #E2E8F0;
        margin-bottom: 16px;
    }
    .metric-note {
        font-size: 13px;
        color: #64748B;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="main-title">Modelo predictivo PETG - Resistencia al impacto</div>
    <div class="subtitle">
    Aplicación interactiva para estimar la resistencia al impacto de probetas PETG impresas por FDM,
    comparar modelos predictivos y visualizar el costo productivo de impresión.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PARÁMETROS FIJOS DE COSTO
# ============================================================

PRECIO_PETG_KG = 20.00          # USD/kg
MASA_PROBETA_G = 6.48           # g/probeta, estimado con volumen y densidad PETG
TARIFA_KWH = 0.09               # USD/kWh
POTENCIA_KW = 0.25              # kW, estimado para Creality K1
COSTO_MAQUINA_H = 0.20          # USD/h
COSTO_OPERADOR_PROBETA = 0.00   # USD/probeta
COSTO_POSTPROCESO = 0.00        # USD/probeta
PORCENTAJE_DESPERDICIO = 0.05   # 5 %


# ============================================================
# CARGA AUTOMÁTICA DE DATOS
# ============================================================

@st.cache_data
def cargar_datos():
    df = pd.read_excel("datos_izod.xlsx")
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
        raise ValueError(f"Faltan columnas en datos_izod.xlsx: {faltantes}")

    df["Tipo_Dato"] = df["Tipo_Dato"].astype(str).str.strip().str.lower()

    return df


try:
    df = cargar_datos()
except FileNotFoundError:
    st.error(
        "No se encontró el archivo datos_izod.xlsx. "
        "Sube ese archivo al repositorio de GitHub junto con app.py."
    )
    st.stop()
except Exception as e:
    st.error(str(e))
    st.stop()


# ============================================================
# VARIABLES DEL MODELO
# ============================================================

X = df[["Temperatura_c", "Altura_capa_m_m", "Velocidad_m_m_s"]]
y = df["Resistencia_Izod_j_m"]


# ============================================================
# MODELOS
# ============================================================

def crear_modelo_mlp():
    return TransformedTargetRegressor(
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


def crear_modelo_svr():
    return TransformedTargetRegressor(
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


modelo_mlp = crear_modelo_mlp()
modelo_svr = crear_modelo_svr()


# ============================================================
# EVALUACIÓN
# ============================================================

@st.cache_data
def evaluar_modelos(X, y):
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

    mlp = crear_modelo_mlp()
    svr = crear_modelo_svr()

    resultados_mlp = cross_validate(
        mlp,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False
    )

    resultados_svr = cross_validate(
        svr,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False
    )

    comparacion = pd.DataFrame({
        "Modelo": ["Red neuronal MLP", "SVR"],
        "R2_validado": [
            np.mean(resultados_mlp["test_R2"]),
            np.mean(resultados_svr["test_R2"])
        ],
        "R2_desviacion": [
            np.std(resultados_mlp["test_R2"]),
            np.std(resultados_svr["test_R2"])
        ],
        "MAE": [
            -np.mean(resultados_mlp["test_MAE"]),
            -np.mean(resultados_svr["test_MAE"])
        ],
        "RMSE": [
            -np.mean(resultados_mlp["test_RMSE"]),
            -np.mean(resultados_svr["test_RMSE"])
        ]
    })

    return comparacion


comparacion_modelos = evaluar_modelos(X, y)

r2_mlp_validado = comparacion_modelos.loc[
    comparacion_modelos["Modelo"] == "Red neuronal MLP",
    "R2_validado"
].values[0]

mae_mlp_validado = comparacion_modelos.loc[
    comparacion_modelos["Modelo"] == "Red neuronal MLP",
    "MAE"
].values[0]

rmse_mlp_validado = comparacion_modelos.loc[
    comparacion_modelos["Modelo"] == "Red neuronal MLP",
    "RMSE"
].values[0]


# ============================================================
# ENTRENAMIENTO FINAL CON TODOS LOS DATOS
# ============================================================

modelo_final_mlp = crear_modelo_mlp()
modelo_final_mlp.fit(X, y)

y_pred_total = modelo_final_mlp.predict(X)

df_predicciones = df.copy()
df_predicciones["Resistencia_predicha_J_m"] = y_pred_total
df_predicciones["Error"] = (
    df_predicciones["Resistencia_Izod_j_m"]
    - df_predicciones["Resistencia_predicha_J_m"]
)
df_predicciones["Error_abs"] = np.abs(df_predicciones["Error"])
df_predicciones["Error_porcentual_abs"] = (
    df_predicciones["Error_abs"] / df_predicciones["Resistencia_Izod_j_m"]
) * 100


# ============================================================
# R2 DEMOSTRATIVO CON 20 PRIMEROS DATOS
# ============================================================

df_20 = df.head(20).copy()

X_20 = df_20[["Temperatura_c", "Altura_capa_m_m", "Velocidad_m_m_s"]]
y_20 = df_20["Resistencia_Izod_j_m"]

modelo_20 = crear_modelo_mlp()
modelo_20.fit(X_20, y_20)
y_20_pred = modelo_20.predict(X_20)

r2_ajuste_20 = r2_score(y_20, y_20_pred)
mae_ajuste_20 = mean_absolute_error(y_20, y_20_pred)
rmse_ajuste_20 = np.sqrt(mean_squared_error(y_20, y_20_pred))


# ============================================================
# COSTOS
# ============================================================

def crear_tabla_costos():
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
        columns=["Codigo_costo", "Tiempo_lote_h", "Cantidad_probetas_lote"]
    )

    df_costos["Tiempo_por_probeta_h"] = (
        df_costos["Tiempo_lote_h"] / df_costos["Cantidad_probetas_lote"]
    )

    df_costos["Masa_por_probeta_g"] = MASA_PROBETA_G
    df_costos["Masa_lote_g"] = (
        df_costos["Masa_por_probeta_g"] * df_costos["Cantidad_probetas_lote"]
    )

    df_costos["Costo_material_USD"] = (
        (df_costos["Masa_por_probeta_g"] / 1000) * PRECIO_PETG_KG
    )

    df_costos["Costo_energia_USD"] = (
        df_costos["Tiempo_por_probeta_h"] * POTENCIA_KW * TARIFA_KWH
    )

    df_costos["Costo_maquina_USD"] = (
        df_costos["Tiempo_por_probeta_h"] * COSTO_MAQUINA_H
    )

    df_costos["Costo_desperdicio_USD"] = (
        df_costos["Costo_material_USD"] * PORCENTAJE_DESPERDICIO
    )

    df_costos["Costo_operador_USD"] = COSTO_OPERADOR_PROBETA
    df_costos["Costo_postproceso_USD"] = COSTO_POSTPROCESO

    df_costos["Costo_total_USD"] = (
        df_costos["Costo_material_USD"]
        + df_costos["Costo_energia_USD"]
        + df_costos["Costo_maquina_USD"]
        + df_costos["Costo_desperdicio_USD"]
        + df_costos["Costo_operador_USD"]
        + df_costos["Costo_postproceso_USD"]
    )

    df_costos["Costo_total_lote_USD"] = (
        df_costos["Costo_total_USD"] * df_costos["Cantidad_probetas_lote"]
    )

    return df_costos


df_costos = crear_tabla_costos()


def asignar_codigo_costo(ensayo):
    ensayo = str(ensayo).strip()

    if "-" in ensayo:
        grupo = ensayo.split("-")[0]
        numero = grupo.replace("A", "")
        return f"L{numero}"

    return ensayo


df_predicciones["Codigo_costo"] = df_predicciones["Ensayo"].apply(asignar_codigo_costo)

df_resultado = df_predicciones.merge(
    df_costos,
    on="Codigo_costo",
    how="left"
)

df_resultado["Eficiencia_resistencia_costo"] = (
    df_resultado["Resistencia_predicha_J_m"] / df_resultado["Costo_total_USD"]
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
# PREDICCIÓN INVERSA
# ============================================================

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
    columns=["Temperatura_c", "Altura_capa_m_m", "Velocidad_m_m_s"]
)

df_busqueda["Resistencia_predicha_J_m"] = modelo_final_mlp.predict(df_busqueda)


# ============================================================
# PESTAÑAS
# ============================================================

tab_inicio, tab_modelo, tab_explorar, tab_graficas, tab_costos, tab_exportar = st.tabs([
    "🏠 Inicio",
    "🧠 Modelo",
    "🎮 Explorador",
    "📊 Gráficas",
    "💵 Costos",
    "⬇️ Exportar"
])


# ============================================================
# TAB INICIO
# ============================================================

with tab_inicio:
    st.markdown("### Resumen general")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Datos usados", df.shape[0])
    c2.metric("Entradas", "3 variables")
    c3.metric("Salida", "Resistencia")
    c4.metric("Material", "PETG")

    st.markdown("#### ¿Qué hace esta app?")
    st.info(
        "La aplicación predice la resistencia al impacto de probetas PETG impresas por FDM "
        "a partir de temperatura de impresión, altura de capa y velocidad. "
        "Además, estima el costo productivo y muestra la relación entre resistencia y costo."
    )

    st.markdown("#### Parámetros de costo asumidos")
    st.dataframe(
        pd.DataFrame({
            "Parámetro": [
                "Precio PETG",
                "Masa por probeta",
                "Tarifa eléctrica",
                "Potencia promedio",
                "Costo de máquina",
                "Desperdicio"
            ],
            "Valor": [
                PRECIO_PETG_KG,
                MASA_PROBETA_G,
                TARIFA_KWH,
                POTENCIA_KW,
                COSTO_MAQUINA_H,
                PORCENTAJE_DESPERDICIO * 100
            ],
            "Unidad": [
                "USD/kg",
                "g",
                "USD/kWh",
                "kW",
                "USD/h",
                "%"
            ]
        }),
        use_container_width=True
    )

    st.markdown("#### Base de datos")
    st.dataframe(df, use_container_width=True)


# ============================================================
# TAB MODELO
# ============================================================

with tab_modelo:
    st.markdown("### Evaluación del modelo")

    st.markdown("#### Resultado principal")

    c1, c2, c3 = st.columns(3)

    c1.metric("R² validado MLP", f"{r2_mlp_validado:.4f}")
    c2.metric("MAE validado", f"{mae_mlp_validado:.4f} J/m")
    c3.metric("RMSE validado", f"{rmse_mlp_validado:.4f} J/m")

    st.markdown("#### R² de ajuste demostrativo vs R² validado")

    r2_comparacion = pd.DataFrame({
        "Tipo de R²": [
            "R² de ajuste con 20 primeros datos",
            "R² validado con validación cruzada"
        ],
        "Valor": [
            r2_ajuste_20,
            r2_mlp_validado
        ],
        "Interpretación": [
            "Mide qué tan bien el modelo se ajusta a datos ya vistos.",
            "Mide mejor la capacidad de predicción en datos no vistos."
        ]
    })

    st.dataframe(r2_comparacion, use_container_width=True)

    fig_r2 = px.bar(
        r2_comparacion,
        x="Tipo de R²",
        y="Valor",
        text="Valor",
        title="Comparación entre R² de ajuste y R² validado",
        color="Tipo de R²"
    )

    fig_r2.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_r2.update_layout(showlegend=False, yaxis_range=[0, 1.05])
    st.plotly_chart(fig_r2, use_container_width=True)

    st.warning(
        "El R² de ajuste puede ser alto porque se calcula sobre datos ya vistos por el modelo. "
        "Para justificar la capacidad predictiva se debe considerar principalmente el R² validado."
    )

    st.markdown("#### Comparación MLP vs SVR")
    st.dataframe(comparacion_modelos, use_container_width=True)

    fig_modelos = px.bar(
        comparacion_modelos,
        x="Modelo",
        y="R2_validado",
        color="Modelo",
        text="R2_validado",
        title="Comparación de modelos mediante R² validado"
    )

    fig_modelos.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_modelos.update_layout(showlegend=False, yaxis_range=[0, 1.05])
    st.plotly_chart(fig_modelos, use_container_width=True)

    st.markdown("#### Arquitectura de la red neuronal")
    st.info(
        "Modelo principal: MLP con 3 entradas, dos capas ocultas de 16 y 8 neuronas, "
        "y una salida correspondiente a la resistencia al impacto predicha."
    )


# ============================================================
# TAB EXPLORADOR
# ============================================================

with tab_explorar:
    st.markdown("### Explorador interactivo")

    st.markdown("Mueve los controles para observar cómo cambia la resistencia al impacto predicha.")

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

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Temperatura", f"{temperatura} °C")
    c2.metric("Altura de capa", f"{altura:.2f} mm")
    c3.metric("Velocidad", f"{velocidad} mm/s")
    c4.metric("Resistencia predicha", f"{prediccion:.2f} J/m")

    st.markdown("---")
    st.markdown("### Predicción inversa")

    izod_objetivo = st.slider(
        "Selecciona una resistencia objetivo (J/m)",
        min_value=14.0,
        max_value=35.0,
        value=25.0,
        step=0.1
    )

    df_busqueda["Diferencia_objetivo"] = np.abs(
        df_busqueda["Resistencia_predicha_J_m"] - izod_objetivo
    )

    recomendaciones = df_busqueda.sort_values(
        by="Diferencia_objetivo",
        ascending=True
    ).head(10)

    st.markdown("#### Combinaciones más cercanas al objetivo")
    st.dataframe(recomendaciones, use_container_width=True)

    mejores_izod = df_busqueda.sort_values(
        by="Resistencia_predicha_J_m",
        ascending=False
    ).head(10)

    st.markdown("#### Top 10 combinaciones con mayor resistencia predicha")
    st.dataframe(mejores_izod, use_container_width=True)


# ============================================================
# TAB GRÁFICAS
# ============================================================

with tab_graficas:
    st.markdown("### Gráficas principales para exposición")

    st.markdown("#### 1. Resistencia real vs resistencia predicha")

    fig_real_pred = px.scatter(
        df_resultado,
        x="Resistencia_Izod_j_m",
        y="Resistencia_predicha_J_m",
        color="Tipo_Dato",
        hover_data=[
            "Ensayo",
            "Temperatura_c",
            "Altura_capa_m_m",
            "Velocidad_m_m_s",
            "Costo_total_USD"
        ],
        title="Comparación entre resistencia real y resistencia predicha",
        labels={
            "Resistencia_Izod_j_m": "Resistencia real (J/m)",
            "Resistencia_predicha_J_m": "Resistencia predicha (J/m)",
            "Tipo_Dato": "Tipo de dato"
        }
    )

    min_val = min(df_resultado["Resistencia_Izod_j_m"].min(), df_resultado["Resistencia_predicha_J_m"].min())
    max_val = max(df_resultado["Resistencia_Izod_j_m"].max(), df_resultado["Resistencia_predicha_J_m"].max())

    fig_real_pred.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Línea ideal"
        )
    )

    st.plotly_chart(fig_real_pred, use_container_width=True)

    st.markdown("#### 2. Error absoluto por ensayo")

    df_errores = df_resultado.sort_values(by="Error_abs", ascending=False)

    fig_error = px.bar(
        df_errores,
        x="Ensayo",
        y="Error_abs",
        color="Tipo_Dato",
        hover_data=[
            "Resistencia_Izod_j_m",
            "Resistencia_predicha_J_m",
            "Temperatura_c",
            "Altura_capa_m_m",
            "Velocidad_m_m_s"
        ],
        title="Error absoluto de predicción por ensayo",
        labels={
            "Error_abs": "Error absoluto (J/m)",
            "Ensayo": "Ensayo"
        }
    )

    st.plotly_chart(fig_error, use_container_width=True)

    st.markdown("#### 3. Importancia de variables")

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
    }).sort_values(by="Importancia_media", ascending=False)

    fig_importancia = px.bar(
        df_importancia,
        x="Variable",
        y="Importancia_media",
        color="Variable",
        title="Importancia de variables en la predicción",
        labels={
            "Importancia_media": "Importancia media"
        }
    )

    st.plotly_chart(fig_importancia, use_container_width=True)

    st.dataframe(df_importancia, use_container_width=True)

    st.markdown("#### 4. Mapa de calor interactivo")

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

    fig_heatmap = go.Figure(
        data=go.Contour(
            z=Z_grid,
            x=temperaturas,
            y=velocidades,
            colorscale="Viridis",
            colorbar=dict(title="Resistencia<br>(J/m)")
        )
    )

    fig_heatmap.update_layout(
        title=f"Mapa de calor de resistencia predicha - altura fija {altura_fija:.2f} mm",
        xaxis_title="Temperatura de impresión (°C)",
        yaxis_title="Velocidad de impresión (mm/s)"
    )

    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.markdown("#### 5. Relación entre resistencia al impacto y costo de impresión")

    fig_res_cost = px.scatter(
        df_resultado,
        x="Costo_total_USD",
        y="Resistencia_predicha_J_m",
        color="Tipo_Dato",
        size="Velocidad_m_m_s",
        hover_data=[
            "Ensayo",
            "Temperatura_c",
            "Altura_capa_m_m",
            "Velocidad_m_m_s",
            "Costo_total_USD",
            "Resistencia_predicha_J_m"
        ],
        title="Relación entre resistencia al impacto predicha y costo de impresión",
        labels={
            "Costo_total_USD": "Costo de impresión por probeta (USD)",
            "Resistencia_predicha_J_m": "Resistencia al impacto predicha (J/m)",
            "Tipo_Dato": "Tipo de dato",
            "Velocidad_m_m_s": "Velocidad"
        }
    )

    st.plotly_chart(fig_res_cost, use_container_width=True)

    st.markdown("#### 6. Eficiencia resistencia/costo")

    df_eficiencia = df_resultado.sort_values(
        by="Eficiencia_resistencia_costo",
        ascending=False
    )

    fig_ef = px.bar(
        df_eficiencia,
        x="Ensayo",
        y="Eficiencia_resistencia_costo",
        color="Tipo_Dato",
        title="Eficiencia: resistencia predicha por dólar de impresión",
        labels={
            "Eficiencia_resistencia_costo": "J/m por USD",
            "Ensayo": "Ensayo"
        }
    )

    st.plotly_chart(fig_ef, use_container_width=True)


# ============================================================
# TAB COSTOS
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

    st.markdown("#### Fórmula usada")

    st.latex(
        r"C_{pieza}=C_{material}+C_{energía}+C_{máquina}+C_{desperdicio}+C_{operador}+C_{postproceso}"
    )

    st.markdown("#### Tabla de costos")

    st.dataframe(df_costos, use_container_width=True)

    st.markdown("#### Desglose promedio del costo")

    fig_comp = px.bar(
        componentes_costo,
        x="Componente",
        y="Costo_promedio_USD",
        color="Componente",
        title="Desglose promedio del costo productivo",
        labels={
            "Costo_promedio_USD": "Costo promedio (USD/probeta)"
        }
    )

    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("#### Costo por ensayo")

    fig_costo = px.bar(
        df_costos,
        x="Codigo_costo",
        y="Costo_total_USD",
        color="Cantidad_probetas_lote",
        title="Costo estimado por probeta",
        labels={
            "Codigo_costo": "Código de impresión",
            "Costo_total_USD": "Costo por probeta (USD)",
            "Cantidad_probetas_lote": "Cantidad/lote"
        }
    )

    st.plotly_chart(fig_costo, use_container_width=True)


# ============================================================
# TAB EXPORTAR
# ============================================================

with tab_exportar:
    st.markdown("### Exportar resultados")

    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Datos_originales", index=False)
        comparacion_modelos.to_excel(writer, sheet_name="Comparacion_modelos", index=False)
        df_resultado.to_excel(writer, sheet_name="Predicciones_costos", index=False)
        df_costos.to_excel(writer, sheet_name="Costos", index=False)
        componentes_costo.to_excel(writer, sheet_name="Componentes_costo", index=False)
        recomendaciones.to_excel(writer, sheet_name="Prediccion_inversa", index=False)
        mejores_izod.to_excel(writer, sheet_name="Mejores_resistencias", index=False)

    st.download_button(
        label="Descargar resultados en Excel",
        data=output.getvalue(),
        file_name="resultados_petg_resistencia_costo.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.markdown("#### Resumen para exposición")

    st.write(
        f"""
        El modelo principal corresponde a una red neuronal artificial MLP con tres variables de entrada:
        temperatura de impresión, altura de capa y velocidad de impresión. La variable de salida es la
        resistencia al impacto predicha.

        La red neuronal utiliza dos capas ocultas de 16 y 8 neuronas. Para evaluar el desempeño se muestran
        dos métricas: un R² de ajuste calculado con los 20 primeros datos y un R² validado mediante validación
        cruzada repetida. El R² de ajuste permite visualizar la capacidad del modelo para representar datos ya
        conocidos, mientras que el R² validado representa mejor la capacidad predictiva frente a datos no vistos.

        R² de ajuste con 20 primeros datos: {r2_ajuste_20:.4f}  
        R² validado MLP: {r2_mlp_validado:.4f}  
        MAE validado MLP: {mae_mlp_validado:.4f} J/m  
        RMSE validado MLP: {rmse_mlp_validado:.4f} J/m  

        Además, se incorpora una estimación de costo productivo por probeta considerando material, energía,
        uso de máquina y desperdicio. Esto permite visualizar la relación entre resistencia al impacto y costo
        de impresión.
        """
    )
