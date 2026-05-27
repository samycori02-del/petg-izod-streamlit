import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


st.set_page_config(
    page_title="Modelo PETG - Izod",
    layout="wide"
)

st.title("Modelo predictivo PETG - Resistencia al impacto Izod")
st.write("Red neuronal MLP para predecir resistencia al impacto Izod en probetas PETG impresas por FDM.")


# ==============================
# 1. CARGA DE DATOS
# ==============================

archivo = st.file_uploader(
    "Sube tu archivo Excel con los datos",
    type=["xlsx"]
)

if archivo is not None:

    df = pd.read_excel(archivo)

    # Limpieza básica
    df["Tipo_Dato"] = df["Tipo_Dato"].astype(str).str.strip().str.lower()

    st.subheader("Datos cargados")
    st.dataframe(df)

    st.write("Filas y columnas:", df.shape)

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


    # ==============================
    # 2. VARIABLES DEL MODELO
    # ==============================

    X = df[["Temperatura_c", "Altura_capa_m_m", "Velocidad_m_m_s"]]
    y = df["Resistencia_Izod_j_m"]


    # ==============================
    # 3. MODELO MLP
    # ==============================

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


    # ==============================
    # 4. VALIDACIÓN CRUZADA
    # ==============================

    st.subheader("Evaluación del modelo")

    if st.button("Entrenar y evaluar modelo"):

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

        resultados_cv = cross_validate(
            modelo_mlp,
            X,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=False
        )

        r2_prom = np.mean(resultados_cv["test_R2"])
        mae_prom = -np.mean(resultados_cv["test_MAE"])
        rmse_prom = -np.mean(resultados_cv["test_RMSE"])

        col1, col2, col3 = st.columns(3)

        col1.metric("R² promedio", f"{r2_prom:.4f}")
        col2.metric("MAE", f"{mae_prom:.4f} J/m")
        col3.metric("RMSE", f"{rmse_prom:.4f} J/m")

        st.session_state["modelo_entrenado"] = modelo_mlp.fit(X, y)
        st.session_state["X"] = X
        st.session_state["y"] = y
        st.session_state["df"] = df

        st.success("Modelo entrenado correctamente.")


    # Si todavía no se entrena, entrenar automáticamente para usar la app
    if "modelo_entrenado" not in st.session_state:
        st.session_state["modelo_entrenado"] = modelo_mlp.fit(X, y)
        st.session_state["X"] = X
        st.session_state["y"] = y
        st.session_state["df"] = df

    modelo_final = st.session_state["modelo_entrenado"]


    # ==============================
    # 5. PREDICCIÓN DIRECTA
    # ==============================

    st.subheader("Predicción directa")

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

    prediccion = modelo_final.predict(nuevo_dato)[0]

    st.success(f"Resistencia Izod predicha: {prediccion:.4f} J/m")


    # ==============================
    # 6. REAL VS PREDICHO
    # ==============================

    st.subheader("Comparación real vs predicho")

    y_pred = modelo_final.predict(X)

    df_pred = df.copy()
    df_pred["Izod_predicho_J_m"] = y_pred
    df_pred["Error_abs"] = abs(
        df_pred["Resistencia_Izod_j_m"] - df_pred["Izod_predicho_J_m"]
    )

    st.dataframe(df_pred)

    fig1, ax1 = plt.subplots(figsize=(7, 6))

    ax1.scatter(
        df_pred["Resistencia_Izod_j_m"],
        df_pred["Izod_predicho_J_m"]
    )

    min_val = min(
        df_pred["Resistencia_Izod_j_m"].min(),
        df_pred["Izod_predicho_J_m"].min()
    )

    max_val = max(
        df_pred["Resistencia_Izod_j_m"].max(),
        df_pred["Izod_predicho_J_m"].max()
    )

    ax1.plot([min_val, max_val], [min_val, max_val])

    ax1.set_xlabel("Resistencia Izod real (J/m)")
    ax1.set_ylabel("Resistencia Izod predicha (J/m)")
    ax1.set_title("Real vs predicho")
    ax1.grid(True)

    st.pyplot(fig1)


    # ==============================
    # 7. MAPA DE CALOR
    # ==============================

    st.subheader("Mapa de calor de resistencia Izod predicha")

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

    Z_pred = modelo_final.predict(datos_malla)
    Z_grid = Z_pred.reshape(T_grid.shape)

    fig2, ax2 = plt.subplots(figsize=(9, 6))

    contorno = ax2.contourf(
        T_grid,
        V_grid,
        Z_grid,
        levels=20,
        cmap="viridis"
    )

    fig2.colorbar(
        contorno,
        ax=ax2,
        label="Resistencia Izod predicha (J/m)"
    )

    ax2.set_xlabel("Temperatura de impresión (°C)")
    ax2.set_ylabel("Velocidad de impresión (mm/s)")
    ax2.set_title(
        f"Mapa de calor - Altura fija = {altura_fija:.2f} mm"
    )

    ax2.grid(alpha=0.3)

    st.pyplot(fig2)


    # ==============================
    # 8. PREDICCIÓN INVERSA
    # ==============================

    st.subheader("Predicción inversa")

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

    df_busqueda["Izod_predicho_J_m"] = modelo_final.predict(df_busqueda)

    df_busqueda["Diferencia_objetivo"] = abs(
        df_busqueda["Izod_predicho_J_m"] - izod_objetivo
    )

    recomendaciones = df_busqueda.sort_values(
        by="Diferencia_objetivo",
        ascending=True
    ).head(10)

    st.write(f"Combinaciones más cercanas a {izod_objetivo:.2f} J/m")
    st.dataframe(recomendaciones)


else:
    st.info("Sube el archivo Excel para iniciar la aplicación.")
