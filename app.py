# ============================================================
# APP STREAMLIT - PETG FDM
# Modelo predictivo de resistencia al impacto + costo productivo
# Red neuronal MLP 3-5-1
# ============================================================

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
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
    .block-container {
        padding-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="main-title">Modelo predictivo PETG - Resistencia al impacto</div>
    <div class="subtitle">
    Aplicación interactiva para estimar la resistencia al impacto y el costo productivo
    de probetas PETG impresas por FDM mediante una red neuronal MLP 3-5-1.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PARÁMETROS FIJOS DE COSTO
# ============================================================

PRECIO_PETG_KG = 20.00          # USD/kg
MASA_PROBETA_G = 6.48           # g/probeta
TARIFA_KWH = 0.09               # USD/kWh
POTENCIA_KW = 0.25              # kW
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
# VARIABLES DE ENTRADA
# ============================================================

X = df[[
    "Temperatura_c",
    "Altura_capa_m_m",
    "Velocidad_m_m_s"
]]

y_resistencia = df["Resistencia_Izod_j_m"]


# ============================================================
# MODELO MLP 3-5-1
# ============================================================

def crear_modelo_mlp_351():
    """
    Red neuronal MLP:
    3 entradas -> 5 neuronas -> 1 salida
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(5,),
            activation="tanh",
            solver="lbfgs",
            alpha=0.01,
            max_iter=10000,
            random_state=42
        ))
    ])


# ============================================================
# MODELO DE RESISTENCIA
# ============================================================

modelo_resistencia = crear_modelo_mlp_351()
modelo_resistencia.fit(X, y_resistencia)

df_resultado = df.copy()
df_resultado["Resistencia_predicha_J_m"] = modelo_resistencia.predict(X)

df_resultado["Error_resistencia"] = (
    df_resultado["Resistencia_Izod_j_m"]
    - df_resultado["Resistencia_predicha_J_m"]
)

df_resultado["Error_abs_resistencia"] = np.abs(
    df_resultado["Error_resistencia"]
)

df_resultado["Error_porcentual_abs_resistencia"] = (
    df_resultado["Error_abs_resistencia"]
    / df_resultado["Resistencia_Izod_j_m"]
) * 100

r2_resistencia = r2_score(
    df_resultado["Resistencia_Izod_j_m"],
    df_resultado["Resistencia_predicha_J_m"]
)

mae_resistencia = mean_absolute_error(
    df_resultado["Resistencia_Izod_j_m"],
    df_resultado["Resistencia_predicha_J_m"]
)

rmse_resistencia = np.sqrt(mean_squared_error(
    df_resultado["Resistencia_Izod_j_m"],
    df_resultado["Resistencia_predicha_J_m"]
))


# ============================================================
# TABLA DE COSTOS CALCULADOS
# ============================================================

def crear_tabla_costos():
    datos_tiempo = [
        # Código, Tiempo_lote_h, Cantidad_probetas_lote

        # L1-L9: lotes de 5 probetas
        ("L1",  1 + 44/60,        5),
        ("L2",  40/60 + 3/3600,   5),
        ("L3",  29/60 + 15/3600,  5),
        ("L4",  1 + 1/60,         5),
        ("L5",  32/60 + 7/3600,   5),
        ("L6",  51/60 + 48/3600,  5),
        ("L7",  42/60 + 1/3600,   5),
        ("L8",  1 + 6/60,         5),
        ("L9",  31/60 + 39/3600,  5),

        # A1-A20: unitarias
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

    df_costos_local = pd.DataFrame(
        datos_tiempo,
        columns=[
            "Codigo_costo",
            "Tiempo_lote_h",
            "Cantidad_probetas_lote"
        ]
    )

    df_costos_local["Tiempo_por_probeta_h"] = (
        df_costos_local["Tiempo_lote_h"]
        / df_costos_local["Cantidad_probetas_lote"]
    )

    df_costos_local["Masa_por_probeta_g"] = MASA_PROBETA_G

    df_costos_local["Masa_lote_g"] = (
        df_costos_local["Masa_por_probeta_g"]
        * df_costos_local["Cantidad_probetas_lote"]
    )

    df_costos_local["Costo_material_USD"] = (
        (df_costos_local["Masa_por_probeta_g"] / 1000)
        * PRECIO_PETG_KG
    )

    df_costos_local["Costo_energia_USD"] = (
        df_costos_local["Tiempo_por_probeta_h"]
        * POTENCIA_KW
        * TARIFA_KWH
    )

    df_costos_local["Costo_maquina_USD"] = (
        df_costos_local["Tiempo_por_probeta_h"]
        * COSTO_MAQUINA_H
    )

    df_costos_local["Costo_desperdicio_USD"] = (
        df_costos_local["Costo_material_USD"]
        * PORCENTAJE_DESPERDICIO
    )

    df_costos_local["Costo_operador_USD"] = COSTO_OPERADOR_PROBETA
    df_costos_local["Costo_postproceso_USD"] = COSTO_POSTPROCESO

    df_costos_local["Costo_total_USD"] = (
        df_costos_local["Costo_material_USD"]
        + df_costos_local["Costo_energia_USD"]
        + df_costos_local["Costo_maquina_USD"]
        + df_costos_local["Costo_desperdicio_USD"]
        + df_costos_local["Costo_operador_USD"]
        + df_costos_local["Costo_postproceso_USD"]
    )

    df_costos_local["Costo_total_lote_USD"] = (
        df_costos_local["Costo_total_USD"]
        * df_costos_local["Cantidad_probetas_lote"]
    )

    return df_costos_local


df_costos = crear_tabla_costos()


# ============================================================
# ASIGNACIÓN DE COSTO A CADA ENSAYO
# ============================================================

def asignar_codigo_costo(ensayo):
    ensayo = str(ensayo).strip()

    # A1-1, A1-2, etc. pertenecen al lote L1
    if "-" in ensayo:
        grupo = ensayo.split("-")[0]
        numero = grupo.replace("A", "")
        return f"L{numero}"

    # A1, A2, ..., A20 son unitarias
    return ensayo


df_resultado["Codigo_costo"] = df_resultado["Ensayo"].apply(asignar_codigo_costo)

df_resultado = df_resultado.merge(
    df_costos,
    on="Codigo_costo",
    how="left"
)

if df_resultado["Costo_total_USD"].isna().any():
    st.warning(
        "Algunos ensayos no tienen costo asignado. Revisa los nombres de la columna Ensayo."
    )


# ============================================================
# MODELO DE COSTO MLP 3-5-1
# ============================================================

X_costo = df_resultado[[
    "Temperatura_c",
    "Altura_capa_m_m",
    "Velocidad_m_m_s"
]]

y_costo = df_resultado["Costo_total_USD"]

modelo_costo = crear_modelo_mlp_351()
modelo_costo.fit(X_costo, y_costo)

df_resultado["Costo_predicho_USD"] = modelo_costo.predict(X_costo)

df_resultado["Error_costo_USD"] = (
    df_resultado["Costo_total_USD"]
    - df_resultado["Costo_predicho_USD"]
)

df_resultado["Error_abs_costo_USD"] = np.abs(
    df_resultado["Error_costo_USD"]
)

r2_costo = r2_score(
    df_resultado["Costo_total_USD"],
    df_resultado["Costo_predicho_USD"]
)

mae_costo = mean_absolute_error(
    df_resultado["Costo_total_USD"],
    df_resultado["Costo_predicho_USD"]
)

rmse_costo = np.sqrt(mean_squared_error(
    df_resultado["Costo_total_USD"],
    df_resultado["Costo_predicho_USD"]
))


# ============================================================
# EFICIENCIA TÉCNICO-PRODUCTIVA
# ============================================================

df_resultado["Eficiencia_resistencia_costo"] = (
    df_resultado["Resistencia_predicha_J_m"]
    / df_resultado["Costo_predicho_USD"]
)


# ============================================================
# COMPONENTES DE COSTO
# ============================================================

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
# PREDICCIÓN INVERSA / BÚSQUEDA DE COMBINACIONES
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
    columns=[
        "Temperatura_c",
        "Altura_capa_m_m",
        "Velocidad_m_m_s"
    ]
)

df_busqueda["Resistencia_predicha_J_m"] = modelo_resistencia.predict(
    df_busqueda[[
        "Temperatura_c",
        "Altura_capa_m_m",
        "Velocidad_m_m_s"
    ]]
)

df_busqueda["Costo_predicho_USD"] = modelo_costo.predict(
    df_busqueda[[
        "Temperatura_c",
        "Altura_capa_m_m",
        "Velocidad_m_m_s"
    ]]
)

df_busqueda["Eficiencia_resistencia_costo"] = (
    df_busqueda["Resistencia_predicha_J_m"]
    / df_busqueda["Costo_predicho_USD"]
)


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
    c2.metric("Arquitectura MLP", "3-5-1")
    c3.metric("Material", "PETG")
    c4.metric("Aplicación", "FDM")

    st.info(
        "Esta aplicación utiliza una red neuronal MLP 3-5-1 para predecir "
        "la resistencia al impacto y el costo productivo de probetas PETG impresas por FDM. "
        "El objetivo es apoyar decisiones de producción considerando desempeño mecánico y costo."
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

    st.markdown("#### Base de datos experimental")
    st.dataframe(df, use_container_width=True)


# ============================================================
# TAB MODELO
# ============================================================

with tab_modelo:
    st.markdown("### Evaluación del modelo MLP 3-5-1")

    st.markdown("#### Métricas de ajuste")

    c1, c2, c3 = st.columns(3)

    c1.metric("R² ajuste resistencia", f"{r2_resistencia:.4f}")
    c2.metric("MAE resistencia", f"{mae_resistencia:.4f} J/m")
    c3.metric("RMSE resistencia", f"{rmse_resistencia:.4f} J/m")

    c4, c5, c6 = st.columns(3)

    c4.metric("R² ajuste costo", f"{r2_costo:.4f}")
    c5.metric("MAE costo", f"{mae_costo:.4f} USD/probeta")
    c6.metric("RMSE costo", f"{rmse_costo:.4f} USD/probeta")

    st.info(
        "El R² mostrado corresponde a R² de ajuste. Esto significa que el modelo se entrena "
        "y se evalúa sobre los datos experimentales disponibles. La arquitectura utilizada es "
        "3 entradas, una capa oculta de 5 neuronas y una salida."
    )

    resumen_modelos = pd.DataFrame({
        "Modelo": [
            "MLP resistencia al impacto",
            "MLP costo productivo"
        ],
        "Arquitectura": [
            "3-5-1",
            "3-5-1"
        ],
        "Tipo_R2": [
            "R2 de ajuste",
            "R2 de ajuste"
        ],
        "R2": [
            r2_resistencia,
            r2_costo
        ],
        "MAE": [
            mae_resistencia,
            mae_costo
        ],
        "RMSE": [
            rmse_resistencia,
            rmse_costo
        ],
        "Unidad_error": [
            "J/m",
            "USD/probeta"
        ]
    })

    st.markdown("#### Resumen de modelos")
    st.dataframe(resumen_modelos, use_container_width=True)

    fig_r2 = px.bar(
        resumen_modelos,
        x="Modelo",
        y="R2",
        color="Modelo",
        text="R2",
        title="R² de ajuste de los modelos MLP 3-5-1",
        labels={
            "R2": "R² de ajuste",
            "Modelo": "Modelo"
        }
    )

    fig_r2.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_r2.update_layout(showlegend=False, yaxis_range=[0, 1.05])
    st.plotly_chart(fig_r2, use_container_width=True)

    st.markdown("#### Arquitectura de la red neuronal")

    st.write(
        """
        La red neuronal utilizada es una MLP con arquitectura **3-5-1**:

        **3 entradas:** temperatura de impresión, altura de capa y velocidad.  
        **5 neuronas:** una capa oculta que aprende relaciones no lineales.  
        **1 salida:** resistencia al impacto o costo productivo, según el modelo entrenado.
        """
    )


# ============================================================
# TAB EXPLORADOR
# ============================================================

with tab_explorar:
    st.markdown("### Explorador interactivo")

    st.markdown(
        "Mueve los controles para observar cómo cambian la resistencia al impacto predicha, "
        "el costo predicho y la eficiencia resistencia/costo."
    )

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

    resistencia_predicha = modelo_resistencia.predict(nuevo_dato)[0]
    costo_predicho = modelo_costo.predict(nuevo_dato)[0]
    eficiencia = resistencia_predicha / costo_predicho if costo_predicho > 0 else np.nan

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Temperatura", f"{temperatura} °C")
    c2.metric("Altura de capa", f"{altura:.2f} mm")
    c3.metric("Velocidad", f"{velocidad} mm/s")
    c4.metric("Resistencia predicha", f"{resistencia_predicha:.2f} J/m")
    c5.metric("Costo predicho", f"{costo_predicho:.4f} USD")

    st.success(
        f"Con estos parámetros, el modelo MLP 3-5-1 estima una resistencia al impacto de "
        f"{resistencia_predicha:.2f} J/m y un costo productivo de "
        f"{costo_predicho:.4f} USD por probeta. "
        f"La eficiencia resistencia/costo es de {eficiencia:.2f} J/m por USD."
    )

    st.markdown("---")
    st.markdown("### Predicción inversa")

    resistencia_objetivo = st.slider(
        "Selecciona una resistencia al impacto objetivo (J/m)",
        min_value=14.0,
        max_value=35.0,
        value=25.0,
        step=0.1
    )

    df_busqueda["Diferencia_objetivo"] = np.abs(
        df_busqueda["Resistencia_predicha_J_m"] - resistencia_objetivo
    )

    recomendaciones = df_busqueda.sort_values(
        by="Diferencia_objetivo",
        ascending=True
    ).head(10)

    st.markdown("#### Combinaciones más cercanas al objetivo")
    st.dataframe(recomendaciones, use_container_width=True)

    mejores_resistencias = df_busqueda.sort_values(
        by="Resistencia_predicha_J_m",
        ascending=False
    ).head(10)

    st.markdown("#### Top 10 combinaciones con mayor resistencia predicha")
    st.dataframe(mejores_resistencias, use_container_width=True)

    mejores_eficiencia = df_busqueda.sort_values(
        by="Eficiencia_resistencia_costo",
        ascending=False
    ).head(10)

    st.markdown("#### Top 10 combinaciones con mejor eficiencia resistencia/costo")
    st.dataframe(mejores_eficiencia, use_container_width=True)


# ============================================================
# TAB GRÁFICAS
# ============================================================

with tab_graficas:
    st.markdown("### Gráficas principales para exposición")

    # --------------------------------------------------------
    # 1. Resistencia real vs predicha
    # --------------------------------------------------------

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
            "Velocidad_m_m_s"
        ],
        title="Comparación entre resistencia real y resistencia predicha",
        labels={
            "Resistencia_Izod_j_m": "Resistencia real (J/m)",
            "Resistencia_predicha_J_m": "Resistencia predicha (J/m)",
            "Tipo_Dato": "Tipo de dato"
        }
    )

    min_res = min(
        df_resultado["Resistencia_Izod_j_m"].min(),
        df_resultado["Resistencia_predicha_J_m"].min()
    )

    max_res = max(
        df_resultado["Resistencia_Izod_j_m"].max(),
        df_resultado["Resistencia_predicha_J_m"].max()
    )

    fig_real_pred.add_trace(
        go.Scatter(
            x=[min_res, max_res],
            y=[min_res, max_res],
            mode="lines",
            name="Línea ideal"
        )
    )

    st.plotly_chart(fig_real_pred, use_container_width=True)

    st.caption(
        "Interpretación: los puntos más cercanos a la línea ideal indican mejores predicciones. "
        "El R² de ajuste de resistencia muestra qué tan bien el modelo reproduce los datos experimentales disponibles."
    )

    # --------------------------------------------------------
    # 2. Costo calculado vs costo predicho
    # --------------------------------------------------------

    st.markdown("#### 2. Costo calculado vs costo predicho")

    fig_costo_pred = px.scatter(
        df_resultado,
        x="Costo_total_USD",
        y="Costo_predicho_USD",
        color="Tipo_Dato",
        hover_data=[
            "Ensayo",
            "Temperatura_c",
            "Altura_capa_m_m",
            "Velocidad_m_m_s",
            "Costo_total_USD",
            "Costo_predicho_USD"
        ],
        title="Comparación entre costo calculado y costo predicho",
        labels={
            "Costo_total_USD": "Costo calculado (USD/probeta)",
            "Costo_predicho_USD": "Costo predicho (USD/probeta)",
            "Tipo_Dato": "Tipo de dato"
        }
    )

    min_costo = min(
        df_resultado["Costo_total_USD"].min(),
        df_resultado["Costo_predicho_USD"].min()
    )

    max_costo = max(
        df_resultado["Costo_total_USD"].max(),
        df_resultado["Costo_predicho_USD"].max()
    )

    fig_costo_pred.add_trace(
        go.Scatter(
            x=[min_costo, max_costo],
            y=[min_costo, max_costo],
            mode="lines",
            name="Línea ideal"
        )
    )

    st.plotly_chart(fig_costo_pred, use_container_width=True)

    st.caption(
        "Interpretación: los puntos cercanos a la línea ideal indican que el modelo predice bien el costo productivo. "
        f"El R² de ajuste del costo es {r2_costo:.4f}, con un MAE de {mae_costo:.4f} USD/probeta."
    )

    # --------------------------------------------------------
    # 3. Relación resistencia predicha vs costo predicho
    # --------------------------------------------------------

    st.markdown("#### 3. Relación entre resistencia al impacto y costo productivo predicho")

    fig_res_cost = px.scatter(
        df_resultado,
        x="Costo_predicho_USD",
        y="Resistencia_predicha_J_m",
        color="Tipo_Dato",
        size="Velocidad_m_m_s",
        hover_data=[
            "Ensayo",
            "Temperatura_c",
            "Altura_capa_m_m",
            "Velocidad_m_m_s",
            "Costo_predicho_USD",
            "Resistencia_predicha_J_m"
        ],
        title="Relación entre resistencia al impacto predicha y costo productivo predicho",
        labels={
            "Costo_predicho_USD": "Costo predicho (USD/probeta)",
            "Resistencia_predicha_J_m": "Resistencia al impacto predicha (J/m)",
            "Tipo_Dato": "Tipo de dato",
            "Velocidad_m_m_s": "Velocidad"
        }
    )

    st.plotly_chart(fig_res_cost, use_container_width=True)

    st.caption(
        "Interpretación: esta gráfica permite identificar combinaciones con alta resistencia y bajo costo. "
        "Desde producción, se busca el mejor equilibrio técnico-productivo."
    )

    # --------------------------------------------------------
    # 4. Error absoluto de resistencia
    # --------------------------------------------------------

    st.markdown("#### 4. Error absoluto de resistencia por ensayo")

    df_errores_res = df_resultado.sort_values(
        by="Error_abs_resistencia",
        ascending=False
    )

    fig_error_res = px.bar(
        df_errores_res,
        x="Ensayo",
        y="Error_abs_resistencia",
        color="Tipo_Dato",
        hover_data=[
            "Resistencia_Izod_j_m",
            "Resistencia_predicha_J_m",
            "Temperatura_c",
            "Altura_capa_m_m",
            "Velocidad_m_m_s"
        ],
        title="Error absoluto de predicción de resistencia por ensayo",
        labels={
            "Error_abs_resistencia": "Error absoluto (J/m)",
            "Ensayo": "Ensayo"
        }
    )

    st.plotly_chart(fig_error_res, use_container_width=True)

    st.caption(
        "Interpretación: las barras más altas indican ensayos donde la diferencia entre el valor experimental "
        "y el valor predicho fue mayor."
    )

    # --------------------------------------------------------
    # 5. Error absoluto de costo
    # --------------------------------------------------------

    st.markdown("#### 5. Error absoluto de costo por ensayo")

    df_errores_costo = df_resultado.sort_values(
        by="Error_abs_costo_USD",
        ascending=False
    )

    fig_error_costo = px.bar(
        df_errores_costo,
        x="Ensayo",
        y="Error_abs_costo_USD",
        color="Tipo_Dato",
        hover_data=[
            "Costo_total_USD",
            "Costo_predicho_USD",
            "Temperatura_c",
            "Altura_capa_m_m",
            "Velocidad_m_m_s"
        ],
        title="Error absoluto de predicción de costo por ensayo",
        labels={
            "Error_abs_costo_USD": "Error absoluto de costo (USD/probeta)",
            "Ensayo": "Ensayo"
        }
    )

    st.plotly_chart(fig_error_costo, use_container_width=True)

    st.caption(
        "Interpretación: aunque el R² del costo es menor que el de resistencia, "
        "el error absoluto está en el orden de centavos o fracciones de centavo por probeta."
    )

    # --------------------------------------------------------
    # 6. Importancia de variables para resistencia
    # --------------------------------------------------------

    st.markdown("#### 6. Importancia de variables en la resistencia")

    importancia_res = permutation_importance(
        modelo_resistencia,
        X,
        y_resistencia,
        n_repeats=30,
        random_state=42,
        scoring="r2"
    )

    df_importancia_res = pd.DataFrame({
        "Variable": X.columns,
        "Importancia_media": importancia_res.importances_mean,
        "Importancia_desviacion": importancia_res.importances_std
    }).sort_values(
        by="Importancia_media",
        ascending=False
    )

    fig_importancia_res = px.bar(
        df_importancia_res,
        x="Variable",
        y="Importancia_media",
        color="Variable",
        title="Importancia de variables en la predicción de resistencia",
        labels={
            "Importancia_media": "Importancia media"
        }
    )

    st.plotly_chart(fig_importancia_res, use_container_width=True)

    st.caption(
        "Interpretación: una variable con mayor importancia tiene mayor influencia en la predicción "
        "de la resistencia al impacto."
    )

    st.dataframe(df_importancia_res, use_container_width=True)

    # --------------------------------------------------------
    # 7. Mapa de calor de resistencia
    # --------------------------------------------------------

    st.markdown("#### 7. Mapa de calor de resistencia predicha")

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

    Z_pred = modelo_resistencia.predict(datos_malla)
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

    st.caption(
        "Interpretación: el color representa la resistencia al impacto predicha. "
        "Permite visualizar zonas de parámetros donde el modelo estima mayor desempeño mecánico."
    )

    # --------------------------------------------------------
    # 8. Eficiencia resistencia/costo
    # --------------------------------------------------------

    st.markdown("#### 8. Eficiencia resistencia/costo")

    df_eficiencia = df_resultado.sort_values(
        by="Eficiencia_resistencia_costo",
        ascending=False
    )

    fig_eficiencia = px.bar(
        df_eficiencia,
        x="Ensayo",
        y="Eficiencia_resistencia_costo",
        color="Tipo_Dato",
        title="Eficiencia: resistencia predicha por dólar de costo predicho",
        labels={
            "Eficiencia_resistencia_costo": "J/m por USD",
            "Ensayo": "Ensayo"
        }
    )

    st.plotly_chart(fig_eficiencia, use_container_width=True)

    st.caption(
        "Interpretación: las barras más altas representan combinaciones con mayor resistencia obtenida "
        "por cada dólar de costo productivo estimado."
    )


# ============================================================
# TAB COSTOS
# ============================================================

with tab_costos:
    st.markdown("### Estimación de costo productivo")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Costo calculado mínimo",
        f"{df_resultado['Costo_total_USD'].min():.4f} USD"
    )

    c2.metric(
        "Costo calculado promedio",
        f"{df_resultado['Costo_total_USD'].mean():.4f} USD"
    )

    c3.metric(
        "Costo calculado máximo",
        f"{df_resultado['Costo_total_USD'].max():.4f} USD"
    )

    c4, c5, c6 = st.columns(3)

    c4.metric(
        "Costo predicho mínimo",
        f"{df_resultado['Costo_predicho_USD'].min():.4f} USD"
    )

    c5.metric(
        "Costo predicho promedio",
        f"{df_resultado['Costo_predicho_USD'].mean():.4f} USD"
    )

    c6.metric(
        "Costo predicho máximo",
        f"{df_resultado['Costo_predicho_USD'].max():.4f} USD"
    )

    st.markdown("#### Fórmula base de costo calculado")

    st.latex(
        r"C_{pieza}=C_{material}+C_{energía}+C_{máquina}+C_{desperdicio}+C_{operador}+C_{postproceso}"
    )

    st.markdown("#### Tabla de costos calculados y predichos")

    st.dataframe(
        df_resultado[[
            "Ensayo",
            "Codigo_costo",
            "Temperatura_c",
            "Altura_capa_m_m",
            "Velocidad_m_m_s",
            "Costo_total_USD",
            "Costo_predicho_USD",
            "Error_costo_USD",
            "Error_abs_costo_USD"
        ]],
        use_container_width=True
    )

    st.markdown("#### Desglose promedio del costo calculado")

    fig_comp = px.bar(
        componentes_costo,
        x="Componente",
        y="Costo_promedio_USD",
        color="Componente",
        title="Desglose promedio del costo productivo calculado",
        labels={
            "Costo_promedio_USD": "Costo promedio (USD/probeta)"
        }
    )

    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("#### Costo calculado vs costo predicho por ensayo")

    df_costo_orden = df_resultado.sort_values("Ensayo").copy()

    fig_costo_comparativo = go.Figure()

    fig_costo_comparativo.add_trace(
        go.Bar(
            x=df_costo_orden["Ensayo"],
            y=df_costo_orden["Costo_total_USD"],
            name="Costo calculado"
        )
    )

    fig_costo_comparativo.add_trace(
        go.Bar(
            x=df_costo_orden["Ensayo"],
            y=df_costo_orden["Costo_predicho_USD"],
            name="Costo predicho"
        )
    )

    fig_costo_comparativo.update_layout(
        barmode="group",
        title="Comparación de costo calculado y costo predicho por ensayo",
        xaxis_title="Ensayo",
        yaxis_title="Costo (USD/probeta)"
    )

    st.plotly_chart(fig_costo_comparativo, use_container_width=True)


# ============================================================
# TAB EXPORTAR
# ============================================================

with tab_exportar:
    st.markdown("### Exportar resultados")

    resumen_modelos = pd.DataFrame({
        "Modelo": [
            "MLP resistencia al impacto",
            "MLP costo productivo"
        ],
        "Arquitectura": [
            "3-5-1",
            "3-5-1"
        ],
        "Tipo_R2": [
            "R2 de ajuste",
            "R2 de ajuste"
        ],
        "R2": [
            r2_resistencia,
            r2_costo
        ],
        "MAE": [
            mae_resistencia,
            mae_costo
        ],
        "RMSE": [
            rmse_resistencia,
            rmse_costo
        ],
        "Unidad_error": [
            "J/m",
            "USD/probeta"
        ]
    })

    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Datos_originales", index=False)
        resumen_modelos.to_excel(writer, sheet_name="Resumen_modelos", index=False)
        df_resultado.to_excel(writer, sheet_name="Predicciones_costos", index=False)
        df_costos.to_excel(writer, sheet_name="Costos_calculados", index=False)
        componentes_costo.to_excel(writer, sheet_name="Componentes_costo", index=False)
        recomendaciones.to_excel(writer, sheet_name="Prediccion_inversa", index=False)
        mejores_resistencias.to_excel(writer, sheet_name="Mejores_resistencias", index=False)
        mejores_eficiencia.to_excel(writer, sheet_name="Mejor_eficiencia", index=False)

    st.download_button(
        label="Descargar resultados en Excel",
        data=output.getvalue(),
        file_name="resultados_petg_mlp_351.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.markdown("#### Resumen para exposición")

    st.write(
        f"""
        El modelo principal corresponde a una red neuronal artificial MLP de arquitectura 3-5-1.
        Esta arquitectura utiliza tres variables de entrada: temperatura de impresión, altura de capa
        y velocidad de impresión. La capa oculta posee cinco neuronas y la salida corresponde a la
        variable predicha.

        Se entrenaron dos modelos con la misma arquitectura:
        uno para predecir la resistencia al impacto y otro para predecir el costo productivo.

        R² de ajuste resistencia al impacto: {r2_resistencia:.4f}  
        MAE resistencia: {mae_resistencia:.4f} J/m  
        RMSE resistencia: {rmse_resistencia:.4f} J/m  

        R² de ajuste costo productivo: {r2_costo:.4f}  
        MAE costo: {mae_costo:.4f} USD/probeta  
        RMSE costo: {rmse_costo:.4f} USD/probeta  

        Desde Ingeniería de la Producción, la herramienta permite relacionar parámetros de impresión,
        desempeño mecánico y costo productivo para apoyar la selección de condiciones de fabricación
        más eficientes.
        """
    )
