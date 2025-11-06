# train_model.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import joblib

# 🔹 Datos de entrenamiento (puedes ampliarlos)
textos = [
    "Factura de compra del cliente",
    "Recibo de pago mensual",
    "Contrato de trabajo firmado",
    "Certificación de EPS del empleado",
    "Documento de identidad escaneado",
    "Informe técnico de laboratorio",
    "Certificado médico",
]

categorias = [
    "Factura",
    "Factura",
    "Contrato",
    "Certificado EPS",
    "Documento",
    "Informe",
    "Certificado EPS",
]

# 🔹 Creamos modelo de pipeline: vectorizador + clasificador
modelo = make_pipeline(TfidfVectorizer(), LogisticRegression())

# 🔹 Entrenamos el modelo
modelo.fit(textos, categorias)

# 🔹 Guardamos el modelo entrenado
joblib.dump(modelo, "model.pkl")

print("✅ Modelo entrenado y guardado como model.pkl")
