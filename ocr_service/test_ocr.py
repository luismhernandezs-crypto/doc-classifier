import pytesseract
from PIL import Image


img = Image.open("test1.png")

texto = pytesseract.image_to_string(img, lang="spa")  # español
print("Texto detectado:")
print(texto)

