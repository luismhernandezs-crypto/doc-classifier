# smav_service/metrics.py
from prometheus_client import Counter, Histogram, Gauge
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ============================================================
# MÉTRICAS SMAV - PREFIJO "smav_" PARA EVITAR COLISIONES
# ============================================================

DOCUMENTS_CLASSIFIED = Counter(
    "smav_documents_classified_total",
    "Número de documentos procesados y clasificados por el SMAV"
)

CATEGORY_COUNT = Counter(
    "smav_category_count_total",
    "Conteo por categoría clasificada",
    ["categoria"]
)

CLASSIFICATION_TIME = Histogram(
    "smav_classification_time_seconds",
    "Tiempo total de clasificación por documento"
)

OCR_ERRORS = Counter(
    "smav_ocr_errors_total",
    "Errores ocurridos durante OCR"
)

# ============================================================
# MÉTRICAS ML (GAUGES)
# ============================================================
ML_ACCURACY = Gauge(
    "smav_ml_accuracy",
    "Exactitud global del modelo"
)

ML_PRECISION = Gauge(
    "smav_ml_precision",
    "Precisión macro promedio"
)

ML_RECALL = Gauge(
    "smav_ml_recall",
    "Recall macro promedio"
)

ML_F1 = Gauge(
    "smav_ml_f1",
    "F1-score macro promedio"
)

# ============================================================
# ACTUALIZACIÓN DE MÉTRICAS DE ML
# ============================================================
def update_ml_metrics(true_labels, predicted_labels):
    """
    - true_labels: etiquetas reales guardadas en DB
    - predicted_labels: predicciones del modelo (text_clean)
    """
    if (
        not true_labels
        or not predicted_labels
        or len(true_labels) != len(predicted_labels)
    ):
        print("ML Metrics skipped: labels vacías o desbalanceadas")
        return

    try:
        acc = accuracy_score(true_labels, predicted_labels)
        prec = precision_score(true_labels, predicted_labels, average="macro", zero_division=0)
        rec = recall_score(true_labels, predicted_labels, average="macro", zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, average="macro", zero_division=0)

        ML_ACCURACY.set(float(acc))
        ML_PRECISION.set(float(prec))
        ML_RECALL.set(float(rec))
        ML_F1.set(float(f1))

        print(f"ML Metrics updated: ACC={acc:.3f}, F1={f1:.3f}")

    except Exception as e:
        print("Error actualizando métricas ML:", e)
