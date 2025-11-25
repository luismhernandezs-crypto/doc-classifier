import os
import random
from faker import Faker
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import LETTER, A4

# Configuración
fake = Faker()
categorias = ["Contratos", "Correspondencia", "Educacion", "Facturas",
              "Legal", "RecursosHumanos", "Salud", "Seguridad"]

# Número de PDFs por categoría
N_PDFS_TRAIN = 200
N_PDFS_TEST = 50

# Rutas
BASE_PATH = "dataset"
TRAIN_PATH = os.path.join(BASE_PATH, "train")
TEST_PATH = os.path.join(BASE_PATH, "test")

# Fuentes y tamaños de letra para variar
FUENTES = ["Helvetica", "Times-Roman", "Courier"]
TAM_LETRA = [10, 12, 14, 16]

def crear_pdf(ruta, nombre_archivo, contenido):
    # Elegir tamaño de página aleatorio
    page_size = random.choice([LETTER, A4])
    c = canvas.Canvas(nombre_archivo, pagesize=page_size)

    # Elegir fuente y tamaño aleatorio
    fuente = random.choice(FUENTES)
    tam = random.choice(TAM_LETRA)
    c.setFont(fuente, tam)

    # Posición inicial aleatoria
    x = random.randint(40, 60)
    y = page_size[1] - random.randint(50, 100)

    textobject = c.beginText(x, y)

    # Insertar cada línea con posible error ortográfico
    for linea in contenido.split("\n"):
        if random.random() < 0.1:  # 10% de probabilidad de error
            linea = "".join(random.sample(linea, len(linea)))
        textobject.textLine(linea)

    c.drawText(textobject)
    c.showPage()
    c.save()
    os.rename(nombre_archivo, os.path.join(ruta, os.path.basename(nombre_archivo)))

# Crear PDFs sintéticos
for categoria in categorias:
    ruta_train = os.path.join(TRAIN_PATH, categoria)
    ruta_test = os.path.join(TEST_PATH, categoria)
    os.makedirs(ruta_train, exist_ok=True)
    os.makedirs(ruta_test, exist_ok=True)

    # PDFs de entrenamiento
    for i in range(N_PDFS_TRAIN):
        contenido = "\n".join(fake.paragraphs(nb=random.randint(5, 10)))
        crear_pdf(ruta_train, f"{categoria}_train_{i+1}.pdf", contenido)

    # PDFs de prueba
    for i in range(N_PDFS_TEST):
        contenido = "\n".join(fake.paragraphs(nb=random.randint(5, 10)))
        crear_pdf(ruta_test, f"{categoria}_test_{i+1}.pdf", contenido)

print("PDFs sintéticos generados correctamente en train/ y test/")
