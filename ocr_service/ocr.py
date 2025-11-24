from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from pdf2image import convert_from_bytes
import pytesseract
import io

app = FastAPI(title="OCR Service - Multi-format (Images & PDF)")

@app.get("/")
def read_root():
    return {"message": "OCR API is running correctly! Supports JPG, PNG, TIFF, BMP, and PDF."}

@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        filename = file.filename.lower()

        if filename.endswith((".jpg", ".jpeg", ".png", ".tiff", ".bmp")):
            image = Image.open(io.BytesIO(file_bytes))
            text = pytesseract.image_to_string(image)

        elif filename.endswith(".pdf"):
            images = convert_from_bytes(file_bytes)
            text = ""
            for i, img in enumerate(images):
                page_text = pytesseract.image_to_string(img)
                text += f"\n--- Página {i + 1} ---\n{page_text}"

        else:
            return JSONResponse(
                content={"error": "Formato no compatible. Solo se aceptan JPG, PNG, TIFF, BMP y PDF."},
                status_code=400
            )

        return JSONResponse(content={"extracted_text": text.strip()})

    except UnidentifiedImageError:
        return JSONResponse(
            content={"error": "El archivo no es una imagen válida."},
            status_code=400
        )
    except Exception as e:
        return JSONResponse(
            content={"error": f"Error procesando el archivo: {str(e)}"},
            status_code=500
        )


