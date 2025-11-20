import os
import shutil
import random

# Ruta del dataset actual (ajusta si hace falta)
DATASET_DIR = "dataset"

# Carpetas de salida
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")

# Crear carpetas destino si no existen
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# Lista de categorías
categorias = [
    d for d in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, d))
    and d not in ["train", "test"]
]

print("\n📁 Categorías detectadas:", categorias)

# Porcentaje test
TEST_RATIO = 0.20

for categoria in categorias:
    print(f"\n📂 Procesando categoría: {categoria}")

    # Paths
    origen = os.path.join(DATASET_DIR, categoria)
    destino_train = os.path.join(TRAIN_DIR, categoria)
    destino_test = os.path.join(TEST_DIR, categoria)

    # Crear subcarpetas
    os.makedirs(destino_train, exist_ok=True)
    os.makedirs(destino_test, exist_ok=True)

    # Obtener archivos
    archivos = [
        f for f in os.listdir(origen)
        if os.path.isfile(os.path.join(origen, f))
    ]

    random.shuffle(archivos)

    total = len(archivos)
    test_count = int(total * TEST_RATIO)
    
    test_files = archivos[:test_count]
    train_files = archivos[test_count:]

    print(f"   Total archivos: {total}")
    print(f"   → Train: {len(train_files)}")
    print(f"   → Test: {len(test_files)}")

    # Mover archivos
    for f in train_files:
        shutil.copy(os.path.join(origen, f), os.path.join(destino_train, f))

    for f in test_files:
        shutil.copy(os.path.join(origen, f), os.path.join(destino_test, f))

print("\n✅ División completada con éxito.")
print("Tu dataset está listo en dataset/train y dataset/test\n")
