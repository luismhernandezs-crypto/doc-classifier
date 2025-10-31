FROM python:3.10-slim

WORKDIR /app
COPY . /app

# Instalar dependencias necesarias
RUN apt-get update && apt-get install -y tesseract-ocr libtesseract-dev poppler-utils

# Instalar paquetes Python
RUN pip install --no-cache-dir fastapi uvicorn pillow pytesseract python-multipart

CMD ["uvicorn", "ocr:app", "--host", "0.0.0.0", "--port", "8000"]

