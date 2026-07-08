import streamlit as st
import pandas as pd
import pickle
import joblib
import os
from datetime import datetime

# ==========================================================
# Cargar modelo, encoders y orden de columnas
# ==========================================================
RUTA_MODELOS = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def cargar_recursos():
    # El modelo se guardó con joblib (igual que en clase, sección Pre-Producción)
    modelo = joblib.load(os.path.join(RUTA_MODELOS, "modelo_final_optimizado.pkl"))
    with open(os.path.join(RUTA_MODELOS, "label_encoders.pkl"), "rb") as f:
        label_encoders = pickle.load(f)
    with open(os.path.join(RUTA_MODELOS, "oneHE.pkl"), "rb") as f:
        oneHE = pickle.load(f)
    with open(os.path.join(RUTA_MODELOS, "ordinalEN.pkl"), "rb") as f:
        ordinalEN = pickle.load(f)
    with open(os.path.join(RUTA_MODELOS, "encoderMMS.pkl"), "rb") as f:
        encoderMMS = pickle.load(f)
    with open(os.path.join(RUTA_MODELOS, "columnas_modelo.pkl"), "rb") as f:
        columnas_modelo = pickle.load(f)
    return modelo, label_encoders, oneHE, ordinalEN, encoderMMS, columnas_modelo


modelo, label_encoders, oneHE, ordinalEN, encoderMMS, columnas_modelo = cargar_recursos()

COLUMNAS_LE = ['Género', 'Pareja', 'Dependientes', 'Servicio de Teléfono', 'Factura electrónica']
COLUMNAS_OE = ['Contrato']
COLUMNAS_OHE = ['Multiples líneas', 'Servicio de Internet', 'Seguridad en línea',
                'Copia de seguridad en línea', 'Protección de dispositivos', 'Soporte técnico',
                'Streaming TV', 'Streaming películas', 'Método de pago']
COLUMNAS_NUM = ['Meses de servicio', 'Cobros mensuales', 'Monto acumulado']

# Métricas finales del modelo (Capítulo 5.1 del informe, evaluadas sobre el conjunto de Test)
METRICAS_MODELO = {
    "Accuracy": 0.6802,
    "Precision": 0.4436,
    "Recall": 0.7995,
    "F1-Score": 0.5706,
    "AUC": 0.8122,
}

IMPORTANCIA_VARIABLES = pd.DataFrame({
    "Variable": ["Contrato", "Servicio de Internet (Fibra óptica)", "Meses de servicio",
                 "Factura electrónica", "Cobros mensuales"],
    "Importancia (|coeficiente|)": [0.4909, 0.2827, 0.2014, 0.1707, 0.1292],
}).set_index("Variable")


def transformar_cliente(cliente: dict) -> pd.DataFrame:
    """Aplica al cliente ingresado la misma secuencia de transformación usada en el entrenamiento
    (Capítulo 2.4) y devuelve una fila lista para el modelo."""
    df_cliente = pd.DataFrame([cliente])

    partes = []

    # Label Encoding
    df_le = pd.DataFrame(index=df_cliente.index)
    for col in COLUMNAS_LE:
        df_le[col] = label_encoders[col].transform(df_cliente[col])
    partes.append(df_le)

    # One-Hot Encoding
    df_ohe = pd.DataFrame(
        oneHE.transform(df_cliente[COLUMNAS_OHE]),
        columns=oneHE.get_feature_names_out(COLUMNAS_OHE),
        index=df_cliente.index
    )
    partes.append(df_ohe)

    # Ordinal Encoding
    df_oe = pd.DataFrame(
        ordinalEN.transform(df_cliente[COLUMNAS_OE]),
        columns=COLUMNAS_OE,
        index=df_cliente.index
    )
    partes.append(df_oe)

    # Min-Max Scaler
    df_mms = pd.DataFrame(
        encoderMMS.transform(df_cliente[COLUMNAS_NUM]),
        columns=COLUMNAS_NUM,
        index=df_cliente.index
    )
    partes.append(df_mms)

    # Ciudadano mayor (ya viene en 0/1)
    partes.append(df_cliente[['Ciudadano mayor']])

    df_final = pd.concat(partes, axis=1)

    # Reordenar EXACTAMENTE igual que en el entrenamiento (crítico porque el modelo es sensible
    # a la posición de las columnas, no solo a sus nombres)
    df_final = df_final.reindex(columns=columnas_modelo, fill_value=0)

    return df_final


# ==========================================================
# Configuración general de la página
# ==========================================================
st.set_page_config(
    page_title="Predicción de Churn - Telco",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "historial" not in st.session_state:
    st.session_state.historial = []

# ==========================================================
# Barra lateral
# ==========================================================
with st.sidebar:
    st.markdown("## 📊 Telco Churn")
    st.caption("Business Predictive Analytics — Grupo 1")
    st.divider()
    st.markdown("**Modelo en producción**")
    st.write("Regresión Logística optimizada")
    st.write("15 variables (seleccionadas con RFECV)")
    st.write("Balanceo: `class_weight='balanced'`")
    st.divider()
    st.markdown("**Predicciones en esta sesión**")
    st.metric("Total", len(st.session_state.historial))
    st.divider()
    st.caption("UPC — Sección 9758")

# ==========================================================
# Encabezado
# ==========================================================
st.title("📊 Predicción de Fuga de Clientes (Churn)")
st.write(
    "Ingresa los datos del cliente para estimar su probabilidad de cancelar el servicio "
    "y anticipar una acción de retención antes de que decida irse."
)

tab_prediccion, tab_metricas, tab_historial = st.tabs(
    ["🔮 Predicción", "📈 Métricas del Modelo", "📜 Historial"]
)

# ==========================================================
# TAB 1: Predicción
# ==========================================================
with tab_prediccion:
    with st.form("formulario_cliente"):

        st.subheader("Datos demográficos")
        col1, col2, col3 = st.columns(3)
        genero = col1.selectbox("Género", ["Femenino", "Masculino"])
        ciudadano_mayor = col2.selectbox("¿Es adulto mayor?", ["No", "Sí"])
        pareja = col3.selectbox("¿Tiene pareja?", ["No", "Sí"])
        dependientes = st.selectbox("¿Tiene dependientes?", ["No", "Sí"])

        st.subheader("Contrato y facturación")
        col1, col2 = st.columns(2)
        contrato = col1.selectbox("Tipo de contrato", ["Mensual", "Anual", "Doble"])
        factura_electronica = col2.selectbox("¿Factura electrónica?", ["No", "Sí"])
        metodo_pago = st.selectbox(
            "Método de pago",
            ["Cheque electrónico", "Cheque enviado por correo", "Transferencia bancaria", "Tarjeta de crédito"]
        )

        st.subheader("Antigüedad y cargos")
        col1, col2, col3 = st.columns(3)
        meses_servicio = col1.number_input("Meses de servicio", min_value=0, max_value=100, value=12)
        cobros_mensuales = col2.number_input("Cobros mensuales ($)", min_value=0.0, max_value=200.0, value=65.0)
        monto_acumulado = col3.number_input("Monto acumulado ($)", min_value=0.0, max_value=10000.0, value=780.0)

        st.subheader("Servicios contratados")
        col1, col2 = st.columns(2)
        servicio_telefono = col1.selectbox("¿Servicio de teléfono?", ["No", "Sí"])
        multiples_lineas_opciones = (
            ["No cuenta con servicio telefónico"] if servicio_telefono == "No" else ["No", "Sí"]
        )
        multiples_lineas = col2.selectbox("¿Múltiples líneas?", multiples_lineas_opciones)

        servicio_internet = st.selectbox("Servicio de Internet", ["DSL", "Fibra óptica", "No"])

        if servicio_internet == "No":
            opciones_extra = ["No cuenta con servicio de internet"]
            seguridad = opciones_extra[0]
            copia_seguridad = opciones_extra[0]
            proteccion = opciones_extra[0]
            soporte = opciones_extra[0]
            streaming_tv = opciones_extra[0]
            streaming_peliculas = opciones_extra[0]
            st.info("Los servicios adicionales de internet no aplican porque el cliente no tiene internet contratado.")
        else:
            col1, col2, col3 = st.columns(3)
            seguridad = col1.selectbox("Seguridad en línea", ["No", "Sí"])
            copia_seguridad = col2.selectbox("Copia de seguridad en línea", ["No", "Sí"])
            proteccion = col3.selectbox("Protección de dispositivos", ["No", "Sí"])
            col1, col2, col3 = st.columns(3)
            soporte = col1.selectbox("Soporte técnico", ["No", "Sí"])
            streaming_tv = col2.selectbox("Streaming TV", ["No", "Sí"])
            streaming_peliculas = col3.selectbox("Streaming películas", ["No", "Sí"])

        enviado = st.form_submit_button("Predecir riesgo de fuga", use_container_width=True)

    if enviado:
        cliente = {
            "Género": genero,
            "Ciudadano mayor": 1 if ciudadano_mayor == "Sí" else 0,
            "Pareja": pareja,
            "Dependientes": dependientes,
            "Meses de servicio": meses_servicio,
            "Servicio de Teléfono": servicio_telefono,
            "Multiples líneas": multiples_lineas,
            "Servicio de Internet": servicio_internet,
            "Seguridad en línea": seguridad,
            "Copia de seguridad en línea": copia_seguridad,
            "Protección de dispositivos": proteccion,
            "Soporte técnico": soporte,
            "Streaming TV": streaming_tv,
            "Streaming películas": streaming_peliculas,
            "Contrato": contrato,
            "Factura electrónica": factura_electronica,
            "Método de pago": metodo_pago,
            "Cobros mensuales": cobros_mensuales,
            "Monto acumulado": monto_acumulado,
        }

        X_nuevo = transformar_cliente(cliente)
        prediccion = modelo.predict(X_nuevo)[0]
        probabilidad = modelo.predict_proba(X_nuevo)[0][1]
        nivel_riesgo = "Alto" if prediccion == 1 else "Bajo"

        # Registrar en el historial de la sesión
        st.session_state.historial.append({
            "Hora": datetime.now().strftime("%H:%M:%S"),
            "Contrato": contrato,
            "Internet": servicio_internet,
            "Meses": meses_servicio,
            "Cobro ($)": cobros_mensuales,
            "Riesgo": nivel_riesgo,
            "Probabilidad": f"{probabilidad:.1%}",
        })

        st.divider()
        st.subheader("Resultado")

        col_resultado, col_gauge = st.columns([2, 1])

        with col_resultado:
            if prediccion == 1:
                st.error(f"⚠️ Cliente en RIESGO de fuga — probabilidad estimada: {probabilidad:.1%}")
                if contrato == "Mensual":
                    st.write("**Recomendación:** ofrecer migración a contrato Anual o Doble con beneficio "
                              "de permanencia; los contratos mensuales son el principal factor de riesgo detectado por el modelo.")
                elif cobros_mensuales > 70:
                    st.write("**Recomendación:** evaluar un descuento o plan más ajustado; el cliente tiene "
                              "cargos mensuales altos, un factor asociado a mayor abandono.")
                else:
                    st.write("**Recomendación:** contactar al cliente de forma proactiva para entender su nivel "
                              "de satisfacción antes de que considere cancelar.")
            else:
                st.success(f"✅ Cliente de BAJO riesgo de fuga — probabilidad estimada: {probabilidad:.1%}")
                st.write("No se requiere una acción de retención inmediata, pero se recomienda mantenerlo "
                          "dentro del monitoreo periódico.")

        with col_gauge:
            st.metric("Probabilidad de Churn", f"{probabilidad:.1%}",
                       delta=f"{probabilidad - 0.266:.1%} vs. promedio general", delta_color="inverse")

        with st.expander("Ver detalle técnico de la predicción"):
            st.write(f"Modelo utilizado: **Regresión Logística optimizada** (15 variables seleccionadas por RFECV)")
            st.write(f"Probabilidad de churn (clase 'Sí'): {probabilidad:.4f}")
            st.dataframe(X_nuevo, use_container_width=True)

# ==========================================================
# TAB 2: Métricas del Modelo
# ==========================================================
with tab_metricas:
    st.subheader("Desempeño del modelo sobre el conjunto de Test")
    st.caption("Estas métricas provienen de la evaluación final del Capítulo 5.1 del informe, "
               "sobre datos que el modelo nunca vio durante el entrenamiento.")

    cols = st.columns(5)
    for col, (nombre, valor) in zip(cols, METRICAS_MODELO.items()):
        col.metric(nombre, f"{valor:.2%}" if nombre != "AUC" else f"{valor:.3f}")

    st.info(
        "El modelo prioriza **Recall** sobre Accuracy: interesa detectar la mayor cantidad posible "
        "de clientes que realmente se van a ir, aunque eso implique señalar como riesgo a algunos "
        "que en realidad se iban a quedar. De cada 10 clientes que efectivamente abandonan, el "
        "modelo identifica correctamente a 8."
    )

    st.divider()
    st.subheader("Variables más influyentes")
    st.caption("Magnitud del coeficiente de cada variable en el modelo final (Regresión Logística).")
    st.bar_chart(IMPORTANCIA_VARIABLES, use_container_width=True)

    st.divider()
    st.subheader("Ficha técnica del modelo")
    ficha = pd.DataFrame({
        "Detalle": [
            "Algoritmo", "Variables utilizadas", "Selección de variables", "Balanceo de clases",
            "Hiperparámetro C", "Conjunto de entrenamiento final"
        ],
        "Valor": [
            "Regresión Logística", "15 de 20 originales", "RFECV (Recursive Feature Elimination + CV)",
            "class_weight='balanced'", "0.01", "Train + Validación (5,625 registros)"
        ]
    }).set_index("Detalle")
    st.table(ficha)

# ==========================================================
# TAB 3: Historial de Predicciones
# ==========================================================
with tab_historial:
    st.subheader("Historial de predicciones de esta sesión")
    st.caption(
        "Este historial se guarda mientras la app esté abierta en tu navegador. Si cierras o "
        "recargas la página, se reinicia — está pensado para revisar varios casos durante la "
        "sustentación, no como registro permanente."
    )

    if len(st.session_state.historial) == 0:
        st.write("Todavía no se ha hecho ninguna predicción. Ve a la pestaña **Predicción** para empezar.")
    else:
        df_historial = pd.DataFrame(st.session_state.historial)
        st.dataframe(df_historial, use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            csv_historial = df_historial.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇️ Descargar historial (CSV)",
                data=csv_historial,
                file_name="historial_predicciones_churn.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col2:
            if st.button("🗑️ Limpiar historial", use_container_width=True):
                st.session_state.historial = []
                st.rerun()
