import joblib
from preprocess import clean_text
import numpy as np

# 1. Cargar el modelo local (el que creemos que es Einstein)
print("Cargando modelo local...")
modelo = joblib.load("model.pkl")

# 2. El texto de prueba (la factura)
texto_sucio = "EMPRESA DE SERVICIOS S.A.S. NIT 800.123.456-1. FACTURA ELECTRÓNICA DE VENTA No. FE-98765. Fecha de Generación: 2025-11-19. Adquirente: LUIS PEREZ. NIT/CC: 1.000.000. Dirección: Calle Falsa 123. DETALLE: 1 Servicio de Consultoría de Software. Valor Unitario: 5.000.000. Subtotal: 5.000.000. IVA 19%: 950.000. TOTAL A PAGAR: 5.950.000. Resolución DIAN Nro 18760000001. Gracias por su compra. CUFE: 384729384729384."

# 3. Limpiar como lo hace el servicio
texto_limpio = clean_text(texto_sucio)
print(f"Texto limpio: {texto_limpio[:50]}...")

# 4. Predecir
probas = modelo.predict_proba([texto_limpio])[0]
confianza = float(np.max(probas))
categoria = modelo.predict([texto_limpio])[0]

print("-" * 30)
print(f"CATEGORIA PREDICHA: {categoria}")
print(f"CONFIANZA: {confianza}")
print("-" * 30)
