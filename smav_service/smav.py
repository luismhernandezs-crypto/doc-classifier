# smav.py
from fastapi import FastAPI, Request
import joblib

app = FastAPI(title="SMAV - Machine Learning Classifier")

# Carga el modelo entrenado
modelo = joblib.load("model.pkl")

@app.get("/")
def root():
    return {"message": "SMAV service is running"}

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    text = data.get("text", "")
    categoria = modelo.predict([text])[0]
    return {"categoria_predicha": categoria}
