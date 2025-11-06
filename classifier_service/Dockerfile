FROM python:3.10-slim

WORKDIR /app

# Copiar los archivos del servicio
COPY . /app

# Instalar dependencias
RUN pip install --no-cache-dir fastapi uvicorn requests psycopg2-binary

# Exponer el puerto del servicio
EXPOSE 8080

# Ejecutar la aplicación
CMD ["uvicorn", "classifier:app", "--host", "0.0.0.0", "--port", "8080"]


