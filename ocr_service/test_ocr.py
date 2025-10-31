import pytesseract
from PIL import Image

# Cargar una imagen de prueba
img = Image.open("test1.png")

# Extraer texto
texto = pytesseract.image_to_string(img, lang="spa")  # español
print("Texto detectado:")
print(texto)

