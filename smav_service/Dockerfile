FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir fastapi uvicorn scikit-learn joblib

EXPOSE 8090

CMD ["uvicorn", "smav:app", "--host", "0.0.0.0", "--port", "8090"]
