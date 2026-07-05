# Predicción de Churn - Telco (Business Predictive Analytics)

Aplicación de predicción de fuga de clientes (Churn) construida con Streamlit.
Proyecto: Business Predictive Analytics - Grupo 1

## Archivos

- `app.py`: código de la aplicación.
- `modelo_final_optimizado.pkl`: modelo de clasificación (Árbol de Decisión optimizado).
- `label_encoders.pkl`, `oneHE.pkl`, `ordinalEN.pkl`, `encoderMMS.pkl`: transformadores usados en el entrenamiento.
- `columnas_modelo.pkl`: orden exacto de columnas requerido por el modelo.
- `requirements.txt`: dependencias necesarias.

## Ejecutar localmente

```
pip install -r requirements.txt
streamlit run app.py
```

## Desplegar en Streamlit Community Cloud

1. Sube este repositorio a GitHub (público o privado).
2. Ve a https://share.streamlit.io e inicia sesión con GitHub.
3. Click en "Create app" -> "Yup, I have an app".
4. Selecciona este repositorio, la rama `main` y el archivo `app.py`.
5. Click en "Deploy".
