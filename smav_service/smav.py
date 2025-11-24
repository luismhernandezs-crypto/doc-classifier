# smav_service/smav.py
import os
import io
import joblib
import numpy as np
import requests
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse, Response
from minio import Minio
from preprocess import clean_text
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from metrics import (
    DOCUMENTS_CLASSIFIED,
    CATEGORY_COUNT,
    CLASSIFICATION_TIME,
    OCR_ERRORS,
    update_ml_metrics
)

app = FastAPI(title="SMAV - Processor & Classifier")

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "model/model.pkl")
modelo = joblib.load(MODEL_PATH)

# ---------------------------------------------------------
# ENVIRONMENT CONFIG
# ---------------------------------------------------------
OCR_URL = os.getenv("OCR_URL", "http://ocr_service:8000/extract-text")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "admin123")
BUCKET_INCOMING = os.getenv("MINIO_BUCKET_INCOMING", "incoming-docs")
BUCKET_CLASSIFIED = os.getenv("MINIO_BUCKET_CLASSIFIED", "classified-docs")

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "doc_classifier")
POSTGRES_USER = os.getenv("POSTGRES_USER", "admin")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "admin123")

DB_CONFIG = {
    "host": POSTGRES_HOST,
    "port": POSTGRES_PORT,
    "database": POSTGRES_DB,
    "user": POSTGRES_USER,
    "password": POSTGRES_PASSWORD,
}

# ---------------------------------------------------------
# MINIO CLIENT
# ---------------------------------------------------------
minio_client = Minio(
    MINIO_ENDPOINT.replace("http://", "").replace("https://", ""),
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False,
)

# Ensure buckets exist
for b in (BUCKET_INCOMING, BUCKET_CLASSIFIED):
    try:
        if not minio_client.bucket_exists(b):
            minio_client.make_bucket(b)
    except Exception as e:
        print(f"Could not ensure bucket {b}: {e}")

# ---------------------------------------------------------
# DATABASE SAVE FUNCTION
# ---------------------------------------------------------
def save_record_db(text, categoria):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS clasificaciones (
                id SERIAL PRIMARY KEY,
                texto TEXT,
                categoria VARCHAR(255),
                fecha TIMESTAMP DEFAULT NOW()
            )
        """)

        cur.execute(
            "INSERT INTO clasificaciones (texto, categoria) VALUES (%s, %s)",
            (text[:10000], categoria),
        )

        conn.commit()
        cur.close()
        conn.close()

    except Exception as e:
        print("DB save error:", e)

# ---------------------------------------------------------
# ROOT ENDPOINT
# ---------------------------------------------------------
@app.get("/")
def root():
    return {"message": "SMAV processor running"}

# ---------------------------------------------------------
# TEXT CLASSIFICATION
# ---------------------------------------------------------
@app.post("/classify-text")
async def classify_text_endpoint(payload: dict):
    text = clean_text(payload.get("text", ""))
    if not text:
        return {"categoria": None, "confianza": 0.0}

    categoria = modelo.predict([text])[0]
    probas = modelo.predict_proba([text])[0]
    confianza = float(np.max(probas))

    return {"categoria": categoria, "confianza": confianza}

# ---------------------------------------------------------
# MODEL DOWNLOAD
# ---------------------------------------------------------
@app.get("/model")
async def get_model():
    model_path = "model.pkl"
    if os.path.exists(model_path):
        return FileResponse(model_path, media_type="application/octet-stream", filename="model.pkl")
    else:
        return Response(content="Modelo no encontrado", status_code=404)

# ---------------------------------------------------------
# PROCESS DOCUMENT
# ---------------------------------------------------------
@app.post("/process-document")
async def process_document(file: UploadFile = File(...)):
    with CLASSIFICATION_TIME.time():

        file_bytes = await file.read()
        filename = file.filename or "documento"

        # Upload to MinIO incoming
        try:
            minio_client.put_object(
                BUCKET_INCOMING,
                filename,
                data=io.BytesIO(file_bytes),
                length=len(file_bytes),
                content_type=file.content_type or "application/octet-stream",
            )
            incoming_minio_path = f"s3://{BUCKET_INCOMING}/{filename}"
        except Exception as e:
            OCR_ERRORS.inc()
            raise HTTPException(status_code=500, detail=f"MinIO upload error: {e}")

        # OCR call
        files = {"file": (filename, file_bytes, file.content_type)}
        try:
            ocr_resp = requests.post(OCR_URL, files=files, timeout=120)
        except Exception as e:
            OCR_ERRORS.inc()
            raise HTTPException(status_code=502, detail=f"OCR connection error: {e}")

        if ocr_resp.status_code != 200:
            OCR_ERRORS.inc()
            raise HTTPException(status_code=ocr_resp.status_code, detail=f"OCR error: {ocr_resp.text}")

        ocr_json = ocr_resp.json()
        extracted_text = ocr_json.get("extracted_text", "")

        # Classification
        text_clean = clean_text(extracted_text)
        if not text_clean:
            categoria = "Desconocido"
            confianza = 0.0
        else:
            categoria = modelo.predict([text_clean])[0]
            probas = modelo.predict_proba([text_clean])[0]
            confianza = float(np.max(probas))

        # Metrics
        DOCUMENTS_CLASSIFIED.inc()
        CATEGORY_COUNT.labels(categoria=categoria).inc()

        # Build TXT result
        contenido = (
            f"=== DOCUMENTO CLASIFICADO ===\n"
            f"Archivo original: {filename}\n"
            f"Categoria: {categoria}\n"
            f"Confianza: {confianza:.4f}\n\n"
            f"--- TEXTO EXTRAIDO ---\n{extracted_text}"
        )

        classified_name = f"{categoria.replace(' ', '_')}_{int(__import__('time').time())}.txt"
        classified_bytes = contenido.encode("utf-8")

        # Upload classified
        try:
            minio_client.put_object(
                BUCKET_CLASSIFIED,
                classified_name,
                data=io.BytesIO(classified_bytes),
                length=len(classified_bytes),
                content_type="text/plain",
            )
            classified_minio_path = f"s3://{BUCKET_CLASSIFIED}/{classified_name}"
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"MinIO upload classified error: {e}")

        # Save DB
        try:
            save_record_db(extracted_text, categoria)
        except Exception as e:
            print("DB save failed:", e)

        return JSONResponse(
            content={
                "status": "success",
                "original_file_minio": incoming_minio_path,
                "classified_file_minio": classified_minio_path,
                "final_category": categoria,
                "smav_confidence": f"{confianza:.4f}",
                "text_length": len(extracted_text),
                "extracted_text": extracted_text,
            }
        )

# ---------------------------------------------------------
# GLOBAL METRICS
# ---------------------------------------------------------
@app.get("/metrics/global")
def get_global_metrics():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # All rows
        cur.execute("""
            SELECT categoria AS true_label, texto
            FROM clasificaciones
            ORDER BY fecha ASC
        """)
        rows = cur.fetchall()

        # Count per category
        cur.execute("""
            SELECT categoria, COUNT(*) AS cantidad
            FROM clasificaciones
            GROUP BY categoria
            ORDER BY cantidad DESC
        """)
        conteo_por_categoria = cur.fetchall()

        # Average text length
        cur.execute("""
            SELECT AVG(LENGTH(texto)) AS promedio_longitud
            FROM clasificaciones
        """)
        promedio_longitud = cur.fetchone()["promedio_longitud"]

        # Last 20
        cur.execute("""
            SELECT id, categoria, fecha, LENGTH(texto) AS longitud
            FROM clasificaciones
            ORDER BY fecha DESC
            LIMIT 20
        """)
        ultimos = cur.fetchall()

        cur.close()
        conn.close()

        # ML METRICS
        true_labels = []
        predicted_labels = []

        if rows:
            true_labels = [r["true_label"] for r in rows]

            for r in rows:
                text_clean = clean_text(r["texto"])
                if text_clean:
                    pred = modelo.predict([text_clean])[0]
                else:
                    pred = "Desconocido"
                predicted_labels.append(pred)

        update_ml_metrics(true_labels, predicted_labels)

        return {
            "total_registros": len(rows),
            "conteo_por_categoria": conteo_por_categoria,
            "promedio_longitud_texto": promedio_longitud,
            "ultimos_registros": ultimos,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB metrics error: {e}")

# ---------------------------------------------------------
# PROMETHEUS METRICS
# ---------------------------------------------------------
@app.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
