import os
import csv
import re
import unicodedata
from fpdf import FPDF
from PIL import Image, ImageDraw, ImageFont

# -------------------
# CONFIGURACIÓN
# -------------------
DATASET_FOLDER = "smav_service/dataset"
OUTPUT_FORMATS = ["pdf", "png", "tiff", "jpg"]

FONT_PATH = None  # Ruta a .ttf si quieres cambiar la fuente
FONT_SIZE = 14
IMAGE_WIDTH = 1240  # px
IMAGE_HEIGHT = 1754  # px (aprox A4)
MARGIN = 50  # px
LINE_SPACING = 4  # px entre líneas

CSV_PATH = os.path.join(DATASET_FOLDER, "dataset_ready.csv")

# -------------------
# FUNCIONES
# -------------------
def clean_text(text):
    text = text.lower()
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    text = re.sub(r"[^a-zA-Z0-9áéíóúñü\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def txt_to_pdf(txt_path, pdf_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    lines = text.split("\n")
    for line in lines:
        pdf.multi_cell(0, 5, line)
    pdf.output(pdf_path)

def txt_to_image(txt_path, image_path, image_format="PNG"):
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    img = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), color="white")
    draw = ImageDraw.Draw(img)

    if FONT_PATH:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    else:
        font = ImageFont.load_default()

    y_text = MARGIN
    for line in text.split("\n"):
        words = line.split()
        current_line = ""
        for word in words:
            test_line = current_line + " " + word if current_line else word
            bbox = draw.textbbox((0,0), test_line, font=font)
            width = bbox[2] - bbox[0]
            if width + 2*MARGIN > IMAGE_WIDTH:
                draw.text((MARGIN, y_text), current_line, fill="black", font=font)
                y_text += FONT_SIZE + LINE_SPACING
                current_line = word
                if y_text > IMAGE_HEIGHT - MARGIN:
                    break
            else:
                current_line = test_line
        if y_text > IMAGE_HEIGHT - MARGIN:
            break
        draw.text((MARGIN, y_text), current_line, fill="black", font=font)
        y_text += FONT_SIZE + LINE_SPACING
        if y_text > IMAGE_HEIGHT - MARGIN:
            break

    # Ajuste para Pillow: jpg → JPEG
    save_format = "JPEG" if image_format.upper() == "JPG" else image_format
    img.save(image_path, format=save_format)

def process_category(category_path, csv_writer):
    for filename in os.listdir(category_path):
        if not filename.endswith(".txt"):
            continue
        txt_path = os.path.join(category_path, filename)
        base_name = os.path.splitext(filename)[0]

        with open(txt_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        clean_txt = clean_text(raw_text)

        for fmt in OUTPUT_FORMATS:
            output_file = os.path.join(category_path, f"{base_name}.{fmt}")
            if os.path.exists(output_file):
                continue
            if fmt == "pdf":
                txt_to_pdf(txt_path, output_file)
            else:
                txt_to_image(txt_path, output_file, fmt.upper())

            # Guardar info en CSV
            csv_writer.writerow([f"{base_name}.{fmt}", os.path.abspath(output_file), os.path.basename(category_path), clean_txt])

def main():
    categories = [d for d in os.listdir(DATASET_FOLDER)
                  if os.path.isdir(os.path.join(DATASET_FOLDER, d))]

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["archivo", "ruta", "categoria", "texto_limpio"])

        for cat in categories:
            cat_path = os.path.join(DATASET_FOLDER, cat)
            print(f"Procesando categoría: {cat}")
            process_category(cat_path, csv_writer)

    print(f"¡Conversión completada! CSV listo en {CSV_PATH}")

if __name__ == "__main__":
    main()
