# train_model.py - Entrenamiento profesional con dataset real
import os
import joblib
import numpy as np
from preprocess import clean_text

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

DATASET_PATH = "dataset"   # carpeta principal


# ============================================================
# 1. Cargar dataset desde /train y /test
# ============================================================
def load_dataset(folder):
    textos = []
    etiquetas = []

    categorias = sorted(os.listdir(folder))

    for categoria in categorias:
        ruta_categoria = os.path.join(folder, categoria)

        if not os.path.isdir(ruta_categoria):
            continue

        for filename in os.listdir(ruta_categoria):
            if not filename.endswith(".txt"):
                continue

            ruta_archivo = os.path.join(ruta_categoria, filename)

            try:
                with open(ruta_archivo, "r", encoding="utf-8") as f:
                    contenido = f.read()
                    contenido = clean_text(contenido)
                    textos.append(contenido)
                    etiquetas.append(categoria)
            except Exception as e:
                print(f"⚠️ Error leyendo {ruta_archivo}: {e}")

    return textos, etiquetas


print("📂 Cargando dataset de entrenamiento...")
train_texts, train_labels = load_dataset(os.path.join(DATASET_PATH, "train"))

print("📂 Cargando dataset de prueba (test)...")
test_texts, test_labels = load_dataset(os.path.join(DATASET_PATH, "test"))

print(f"✔ Entrenamiento: {len(train_texts)} documentos")
print(f"✔ Test: {len(test_texts)} documentos")


# ============================================================
# 2. Modelo Machine Learning Profesional
# ============================================================
modelo = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=12000,
        ngram_range=(1, 2),   # unigrams + bigrams
        stop_words=None
    )),
    ("clf", LogisticRegression(
        max_iter=3000,
        n_jobs=-1,
        class_weight="balanced"  # importantísimo para datasets desbalanceados
    ))
])


# ============================================================
# 3. Entrenamiento del modelo
# ============================================================
print("🚀 Entrenando modelo...")
modelo.fit(train_texts, train_labels)


# ============================================================
# 4. Evaluación en TRAIN
# ============================================================
print("\n📊 METRICAS (TRAIN SET)")
train_pred = modelo.predict(train_texts)
print(classification_report(train_labels, train_pred))


# ============================================================
# 5. Evaluación en TEST (REAL)
# ============================================================
print("\n📊 METRICAS (TEST SET)")
test_pred = modelo.predict(test_texts)

print("Accuracy:", accuracy_score(test_labels, test_pred))
print(classification_report(test_labels, test_pred))

print("\n🧩 Matriz de confusión:")
print(confusion_matrix(test_labels, test_pred))


# ============================================================
# 6. Guardar modelo final
# ============================================================
joblib.dump(modelo, "model.pkl")
print("\n✅ Modelo final entrenado y guardado como model.pkl")
