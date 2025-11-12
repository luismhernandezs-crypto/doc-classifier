# train_model.py - versión empresarial corregida y funcional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import joblib

# ===============================
#  Stopwords personalizadas en español
# ===============================
spanish_stopwords = list(text.ENGLISH_STOP_WORDS.union({
    "de", "la", "que", "el", "en", "y", "a", "los", "se", "del", "las", "por", "un",
    "para", "con", "no", "una", "su", "al", "lo", "como", "más", "pero", "sus", "le",
    "ya", "o", "fue", "este", "ha", "sí", "porque", "esta", "son", "entre", "cuando",
    "muy", "sin", "sobre", "también", "me", "hasta", "hay", "donde", "quien", "desde",
    "todo", "nos", "durante", "todos", "uno", "les", "ni", "contra", "otros", "ese",
    "esa", "eso", "está", "han", "ser", "tiene", "cada", "puede", "empresa"
}))

# ===============================
#  Textos de entrenamiento
# ===============================
textos = [
    # --- Contabilidad ---
    "Factura de compra del proveedor",
    "Recibo de pago mensual al empleado",
    "Informe financiero trimestral de ingresos",
    "Comprobante de egreso contable",
    "Pago de nómina del mes de septiembre",

    # --- Recursos Humanos ---
    "Contrato laboral firmado entre la empresa y el trabajador",
    "Certificado laboral del empleado Juan Pérez",
    "Permiso de vacaciones aprobado por recursos humanos",
    "Carta de renuncia voluntaria",
    "Hoja de vida de candidato a empleo",

    # --- Legal ---
    "Contrato de arrendamiento del local comercial",
    "Documento notariado con firma y sello",
    "Póliza de seguros corporativa",
    "Cláusulas legales del contrato de confidencialidad",

    # --- Educación ---
    "Diploma de grado de ingeniería de sistemas",
    "Certificado de curso de capacitación empresarial",
    "Constancia de asistencia a seminario académico",
    "Acta de evaluación de desempeño estudiantil",

    # --- Salud ---
    "Certificado EPS del empleado vigente",
    "Informe médico de incapacidad laboral",
    "Remisión de exámenes clínicos de laboratorio",

    # --- Seguridad y Cumplimiento ---
    "Política de tratamiento de datos personales",
    "Informe de auditoría interna de seguridad",
    "Documento confidencial interno de la compañía",
    "Reporte de incidente de ciberseguridad",

    # --- Correspondencia ---
    "Carta formal dirigida al gerente general",
    "Comunicado interno sobre cambios organizacionales",
    "Memorando enviado al departamento contable",
]

categorias = [
    # Contabilidad
    "Factura", "Recibo de Pago", "Informe Financiero", "Comprobante", "Pago de Nómina",
    # Recursos Humanos
    "Contrato Laboral", "Certificado Laboral", "Permiso", "Carta de Renuncia", "Hoja de Vida",
    # Legal
    "Contrato", "Documento Notariado", "Póliza", "Contrato de Confidencialidad",
    # Educación
    "Diploma", "Certificado de Curso", "Documento Educativo", "Acta Académica",
    # Salud
    "Certificado EPS", "Incapacidad Médica", "Informe Clínico",
    # Seguridad y Cumplimiento
    "Política de Datos", "Informe de Auditoría", "Documento Confidencial", "Reporte de Seguridad",
    # Correspondencia
    "Carta", "Comunicado Interno", "Memorando"
]

# ===============================
# Entrenamiento del modelo
# ===============================
modelo = make_pipeline(
    TfidfVectorizer(stop_words=spanish_stopwords, max_features=5000),
    LogisticRegression(max_iter=500)
)

modelo.fit(textos, categorias)

# ===============================
#  Guardar modelo
# ===============================
joblib.dump(modelo, "model.pkl")
print("✅ Modelo empresarial entrenado y guardado como model.pkl")

