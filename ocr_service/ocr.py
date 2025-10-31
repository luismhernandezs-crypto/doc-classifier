
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import pytesseract
import io

app = FastAPI(title="OCR Service")

@app.get("/")
def read_root():
    return {"message": "OCR API is running correctly!"}

@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):
    # Lee la imagen subida
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    # Extrae texto con Tesseract
    text = pytesseract.image_to_string(image)

    return {"extracted_text": text.strip()}

