from fastapi import FastAPI, Form, UploadFile, File, Request
from fastapi.responses import HTMLResponse
import requests

app = FastAPI(title="Frontend OCR Interface")

# Página principal con formulario
@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
        <head>
            <title>OCR Web Interface</title>
        </head>
        <body style="font-family: Arial; margin: 40px;">
            <h2>Subir imagen para OCR</h2>
            <form action="/upload" enctype="multipart/form-data" method="post">
                <input type="file" name="file" accept="image/*" required>
                <button type="submit">Procesar imagen</button>
            </form>
        </body>
    </html>
    """

# Procesar la imagen subiendo a OCR
@app.post("/upload", response_class=HTMLResponse)
async def upload(file: UploadFile = File(...)):
    ocr_url = "http://ocr_service:8000/extract-text"  # endpoint OCR

    files = {"file": (file.filename, await file.read(), file.content_type)}
    response = requests.post(ocr_url, files=files)

    if response.status_code == 200:
        text = response.json().get("extracted_text", "")
        return f"""
        <html>
            <body style="font-family: Arial; margin: 40px;">
                <h2>Texto extraído:</h2>
                <pre style="background-color:#f5f5f5; padding:10px;">{text}</pre>
                <a href="/">Volver</a>
            </body>
        </html>
        """
    else:
        return f"<p>Error procesando imagen. Código: {response.status_code}</p>"
