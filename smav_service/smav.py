# smav.py - Servicio de Clasificación SMAV
from fastapi import FastAPI, Request
import joblib
import numpy as np
from preprocess import clean_text

app = FastAPI(title="SMAV - Machine Learning Classifier")

# Cargar modelo entrenado
modelo = joblib.load("model.pkl")

@app.get("/")
def root():
    return {"message": "SMAV service is running"}

@app.post("/classify-text")
async def classify_text(request: Request):
    data = await request.json()
    text = clean_text(data.get("text", ""))

    if not text:
        return {
            "categoria": None,
            "confianza": 0.0
        }

    # Predicción real
    categoria = modelo.predict([text])[0]
    probas = modelo.predict_proba([text])[0]
    confianza = float(np.max(probas))

    # Filtro de baja confianza
    if confianza < 0.55:
        return {
            "categoria": None,
            "confianza": confianza
        }

    return {
        "categoria": categoria,
        "confianza": confianza
    }
