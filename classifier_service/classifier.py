# classifier.py
from fastapi import FastAPI, Request
from datetime import datetime
import requests
import psycopg2
from psycopg2 import OperationalError

app = FastAPI(title="Classifier Service - Integración SMAV + PostgreSQL")


SMAV_URL = "http://smav_service:8090/predict"


DB_CONFIG = {
    "host": "postgres",
    "port": "5432",
    "database": "doc_classifier",
    "user": "admin",
    "password": "admin123"
}


def init_db():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS clasificaciones (
                id SERIAL PRIMARY KEY,
                texto TEXT,
                categoria VARCHAR(255),
                fecha TIMESTAMP
            )
        """)
        conn.commit()
        cursor.close()
        conn.close()
        print(" Tabla 'clasificaciones' verificada o creada correctamente.")
    except Exception as e:
        print(f" Error al inicializar la base de datos: {e}")

def save_to_db(text: str, categoria: str):

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO clasificaciones (texto, categoria, fecha)
            VALUES (%s, %s, %s)
        """, (text, categoria, datetime.now()))
        conn.commit()
        cursor.close()
        conn.close()
        print(f" Registro guardado correctamente: {categoria}")
    except OperationalError as e:
        print(f" Error de conexión con PostgreSQL: {e}")
    except Exception as e:
        print(f" Error general al guardar en PostgreSQL: {e}")

@app.on_event("startup")
def startup_event():

    init_db()


@app.get("/")
def root():
    return {"message": " Classifier API corriendo e integrada con SMAV y PostgreSQL"}

@app.post("/classify-text")
async def classify_text(request: Request):

    data = await request.json()
    text = data.get("text", "").strip()

    if not text:
        return {"error": " No se recibió texto para clasificar"}

    try:
        # Enviar texto al servicio SMAV
        response = requests.post(SMAV_URL, json={"text": text})

        if response.status_code == 200:
            result = response.json()
            categoria = result.get("categoria_predicha", "Desconocido")


            save_to_db(text, categoria)

            print(f" Clasificación exitosa → {categoria}")
            return {"texto": text[:150], "categoria": categoria}
        else:
            error_msg = f" SMAV devolvió error HTTP {response.status_code}"
            print(error_msg)
            return {"error": error_msg}

    except Exception as e:
        print(f" Error comunicando con SMAV: {e}")
        return {"error": f"No se pudo conectar con SMAV: {e}"}

