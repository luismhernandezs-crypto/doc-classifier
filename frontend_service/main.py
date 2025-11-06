from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import HTMLResponse
import requests
import psycopg2

app = FastAPI(title="OCR + Classifier Frontend")

# URLs internas dentro de la red Docker
OCR_URL = "http://ocr_service:8000/extract-text"
CLASSIFIER_URL = "http://classifier_service:8080/classify-text"

# Configuración de conexión con PostgreSQL
DB_CONFIG = {
    "host": "postgres",
    "port": "5432",
    "database": "doc_classifier",
    "user": "admin",
    "password": "admin123"
}

# Página principal con formulario
@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
        <head>
            <title>OCR Web Interface</title>
            <style>
                body { font-family: Arial; margin: 40px; background-color: #f9fafb; }
                h2 { color: #333; }
                form { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
                button { background-color: #007bff; color: white; border: none; padding: 10px 15px; border-radius: 5px; cursor: pointer; }
                button:hover { background-color: #0056b3; }
                a { text-decoration: none; color: #007bff; }
            </style>
        </head>
        <body>
            <h2>Subir imagen para OCR</h2>
            <form action="/upload" enctype="multipart/form-data" method="post">
                <input type="file" name="file" accept="image/*" required><br><br>
                <button type="submit">Procesar imagen</button>
            </form>
            <br>
            <a href="/history">📜 Ver historial de clasificaciones</a>
        </body>
    </html>
    """

# Procesar la imagen enviándola al OCR y al clasificador
@app.post("/upload", response_class=HTMLResponse)
async def upload(file: UploadFile = File(...)):
    files = {"file": (file.filename, await file.read(), file.content_type)}
    ocr_response = requests.post(OCR_URL, files=files)

    if ocr_response.status_code == 200:
        extracted_text = ocr_response.json().get("extracted_text", "")

        # Enviar texto al clasificador
        classifier_response = requests.post(CLASSIFIER_URL, json={"text": extracted_text})
        if classifier_response.status_code == 200:
            category = classifier_response.json().get("categoria", "Desconocido")

            return f"""
            <html>
                <body style="font-family: Arial; margin: 40px;">
                    <h2>Resultado del OCR</h2>
                    <pre style="background-color:#f5f5f5; padding:10px;">{extracted_text}</pre>
                    <h3>Categoría detectada: <span style="color:green;">{category}</span></h3>
                    <a href="/">Volver</a> | <a href="/history">Ver historial</a>
                </body>
            </html>
            """
        else:
            return f"<p>Error clasificando texto. Código: {classifier_response.status_code}</p>"
    else:
        return f"<p>Error procesando imagen. Código: {ocr_response.status_code}</p>"

# 🧾 Historial de clasificaciones
@app.get("/history", response_class=HTMLResponse)
def history():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT texto, categoria, fecha FROM clasificaciones ORDER BY fecha DESC;")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        # Construir tabla HTML
        rows_html = "".join(
            f"<tr><td>{texto[:60]}...</td><td>{categoria}</td><td>{fecha}</td></tr>"
            for texto, categoria, fecha in rows
        )

        return f"""
        <html>
            <head>
                <title>Historial de Clasificaciones</title>
                <style>
                    body {{ font-family: Arial; margin: 40px; background-color: #f9fafb; }}
                    table {{ width: 100%; border-collapse: collapse; background: white; }}
                    th, td {{ border: 1px solid #ccc; padding: 10px; text-align: left; }}
                    th {{ background-color: #007bff; color: white; }}
                    tr:nth-child(even) {{ background-color: #f2f2f2; }}
                    a {{ text-decoration: none; color: #007bff; }}
                </style>
            </head>
            <body>
                <h2>📜 Historial de Clasificaciones</h2>
                <table>
                    <tr><th>Texto</th><th>Categoría</th><th>Fecha</th></tr>
                    {rows_html if rows_html else "<tr><td colspan='3'>No hay registros aún.</td></tr>"}
                </table>
                <br><a href="/">⬅ Volver al inicio</a>
            </body>
        </html>
        """
    except Exception as e:
        return f"<p>Error al conectar con la base de datos: {e}</p>"

