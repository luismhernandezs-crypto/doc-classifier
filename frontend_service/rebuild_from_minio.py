#!/usr/bin/env python3
# rebuild_from_minio.py
import os
import io
import time
import requests
import psycopg2
from minio import Minio
from psycopg2.extras import RealDictCursor
from preprocess import clean_text

MINIO_ENDPOINT = os.getenv("MINIO_HOST", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ROOT_USER", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD", "admin123")
BUCKET = os.getenv("MINIO_BUCKET_INCOMING", "incoming-docs")

OCR_URL = os.getenv("OCR_URL", "http://ocr_service:8000/extract-text")
SMAV_PROCESS_URL = os.getenv("SMAV_PROCESS_URL", "http://smav_service:8000/process-document")
CLASSIFIER_URL = os.getenv("CLASSIFIER_URL", "http://classifier_service:8080/classify-text")

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "postgres"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
    "database": os.getenv("POSTGRES_DB", "doc_classifier"),
    "user": os.getenv("POSTGRES_USER", "admin"),
    "password": os.getenv("POSTGRES_PASSWORD", "admin123"),
}

minio_client = Minio(
    MINIO_ENDPOINT.replace("http://", "").replace("https://", ""),
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

def get_db_conn():
    return psycopg2.connect(**DB_CONFIG)

def ensure_table():
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS clasificaciones (
            id SERIAL PRIMARY KEY,
            filename TEXT,
            texto TEXT,
            categoria VARCHAR(255),
            fecha TIMESTAMP DEFAULT NOW(),
            username VARCHAR(100),
            smav_pred VARCHAR(255),
            classifier_pred VARCHAR(255),
            ground_truth VARCHAR(255),
            confidence NUMERIC
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

def process_object(obj_name):
    print("Procesando:", obj_name)
    data = minio_client.get_object(BUCKET, obj_name)
    content = data.read()
    data.close()

    try:
        resp = requests.post(OCR_URL, files={"file": (obj_name, content)}, timeout=60)
        if resp.ok:
            ocr_json = resp.json()
            texto = ocr_json.get("extracted_text", "")
        else:
            print("OCR failed:", resp.status_code)
            texto = ""
    except Exception as e:
        print("OCR exception:", e)
        texto = ""

    smav_categoria = None
    smav_conf = None
    try:
        resp = requests.post("http://smav_service:8000/classify-text", json={"text": texto}, timeout=30)
        if resp.ok:
            j = resp.json()
            smav_categoria = j.get("categoria") or None
            smav_conf = j.get("confianza") or j.get("confidence") or None
        else:
            print("SMAV classify failed:", resp.status_code)
    except Exception as e:
        print("SMAV classify exception:", e)

    classifier_pred = None
    try:
        if texto:
            resp = requests.post(CLASSIFIER_URL, json={"text": texto}, timeout=30)
            if resp.ok:
                j = resp.json()
                classifier_pred = j.get("categoria") or j.get("category") or None
    except Exception as e:
        print("Classifier exception:", e)

    try:
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO clasificaciones
            (filename, texto, categoria, fecha, smav_pred, classifier_pred, ground_truth, confidence)
            VALUES (%s, %s, %s, NOW(), %s, %s, %s, %s)
        """, (obj_name, texto[:10000], smav_categoria, smav_categoria, classifier_pred, classifier_pred, smav_conf))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print("DB insert failed:", e)

def main():
    ensure_table()
    for obj in minio_client.list_objects(BUCKET, recursive=True):
        try:
            process_object(obj.object_name)
            time.sleep(0.2)
        except Exception as e:
            print("Error processing object", obj.object_name, e)

if __name__ == "__main__":
    main()
