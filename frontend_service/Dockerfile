FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install fastapi uvicorn requests python-multipart

EXPOSE 5000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
