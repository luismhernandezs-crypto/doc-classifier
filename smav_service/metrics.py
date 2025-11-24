# metrics.py (smav_service)
from prometheus_client import start_http_server, Counter, Gauge, Histogram

# Contadores
DOCUMENTS_CLASSIFIED = Counter(
    "smav_documents_classified_total",
    "Total de documentos clasificados por SMAV"
)

OCR_ERRORS = Counter(
    "smav_ocr_errors_total",
    "Errores ocurridos durante el OCR"
)

CATEGORY_COUNT = Counter(
    "smav_category_total",
    "Cantidad de documentos clasificados por categoría",
    ["categoria"]
)

# Tiempo de clasificación
CLASSIFICATION_TIME = Histogram(
    "smav_classification_seconds",
    "Tiempo que tarda en clasificar un documento"
)

# Métricas de desempeño ML
ACCURACY = Gauge("smav_model_accuracy", "Accuracy del modelo")
F1_SCORE = Gauge("smav_model_f1_score", "F1 Score del modelo")
PRECISION = Gauge("smav_model_precision", "Precision del modelo")
RECALL = Gauge("smav_model_recall", "Recall del modelo")




def update_ml_metrics(true_labels, predicted_labels):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    ACCURACY.set(accuracy_score(true_labels, predicted_labels))
    F1_SCORE.set(f1_score(true_labels, predicted_labels, average="macro"))
    PRECISION.set(precision_score(true_labels, predicted_labels, average="macro"))
    RECALL.set(recall_score(true_labels, predicted_labels, average="macro"))


