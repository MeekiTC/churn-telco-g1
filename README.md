# Predicción de Churn - Telco (Business Predictive Analytics)

Aplicación de predicción de fuga de clientes (Churn) construida con Streamlit.
Proyecto: Business Predictive Analytics - Grupo 1

## Estructura de la app (3 pestañas)

- **🔮 Predicción**: formulario para ingresar un cliente nuevo y obtener su probabilidad de churn, con recomendación de acción.
- **📈 Métricas del Modelo**: tarjetas con Accuracy, Precision, Recall, F1 y AUC del modelo final (Capítulo 5.1 del informe), gráfico de importancia de variables y ficha técnica.
- **📜 Historial**: registro de las predicciones hechas durante la sesión, con opción de descargar como CSV o limpiar.

## Archivos

- `app.py`: código de la aplicación.
- `.streamlit/config.toml`: tema de colores de la app.
- `modelo_final_optimizado.pkl`: modelo de clasificación (Regresión Logística optimizada, 15 variables).
- `label_encoders.pkl`, `oneHE.pkl`, `ordinalEN.pkl`, `encoderMMS.pkl`: transformadores usados en el entrenamiento.
- `columnas_modelo.pkl`: orden exacto de columnas requerido por el modelo.
- `requirements.txt`: dependencias necesarias.

## Ejecutar localmente

```
pip install -r requirements.txt
streamlit run app.py
```

## Desplegar en Streamlit Community Cloud

1. Sube este repositorio a GitHub (público o privado), incluyendo la carpeta `.streamlit` con `config.toml`.
2. Ve a https://share.streamlit.io e inicia sesión con GitHub.
3. Click en "Create app" -> "Yup, I have an app".
4. Selecciona este repositorio, la rama `main` y el archivo `app.py`.
5. Click en "Deploy".
