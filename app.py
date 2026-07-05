import streamlit as st
import pandas as pd
import pickle
import joblib
import os

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

    # Reordenar EXACTAMENTE igual que en el entrenamiento (crítico para modelos como Árbol de Decisión / RF / XGBoost)
    df_final = df_final.reindex(columns=columnas_modelo, fill_value=0)

    return df_final


# ==========================================================
# Interfaz
# ==========================================================
st.set_page_config(page_title="Predicción de Churn - Telco", page_icon="📊", layout="centered")

st.title("📊 Predicción de Fuga de Clientes (Churn)")
st.caption("Proyecto de Business Predictive Analytics — Telco Customer Churn")
st.write(
    "Ingresa los datos del cliente para estimar su probabilidad de cancelar el servicio "
    "y anticipar una acción de retención antes de que decida irse."
)

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

    enviado = st.form_submit_button("Predecir riesgo de fuga")

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

    st.divider()
    st.subheader("Resultado")

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

    with st.expander("Ver detalle técnico de la predicción"):
        st.write(f"Modelo utilizado: **Regresión Logística optimizada** (15 variables seleccionadas por RFECV)")
        st.write(f"Probabilidad de churn (clase 'Sí'): {probabilidad:.4f}")
        st.dataframe(X_nuevo)
