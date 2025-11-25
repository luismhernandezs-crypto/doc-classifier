import os
from pdf2image import convert_from_path
from PIL import Image
import pytesseract

BASE_PATH = "dataset"

def pdf_a_txt(carpeta):
    categorias = [d for d in os.listdir(carpeta) if os.path.isdir(os.path.join(carpeta, d))]
    for cat in categorias:
        ruta_cat = os.path.join(carpeta, cat)
        for archivo in os.listdir(ruta_cat):
            if archivo.lower().endswith(".pdf"):
                pdf_path = os.path.join(ruta_cat, archivo)
                txt_path = os.path.join(ruta_cat, os.path.splitext(archivo)[0] + ".txt")
                try:
                    imagenes = convert_from_path(pdf_path)
                    texto_completo = ""
                    for i, img in enumerate(imagenes):
                        texto_completo += f"\n--- Página {i+1} ---\n"
                        texto_completo += pytesseract.image_to_string(img)

                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(texto_completo.strip())
                    print(f"Procesado {archivo} → {txt_path}")
                except Exception as e:
                    print(f"Error procesando {archivo}: {e}")

# Ejecutar para train y test
for subset in ["train", "test"]:
    print(f"Procesando {subset.upper()}...")
    pdf_a_txt(os.path.join(BASE_PATH, subset))

print("PDFs convertidos a TXT con OCR correctamente")
