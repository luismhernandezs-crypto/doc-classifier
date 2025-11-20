import re
import unicodedata

def clean_text(text):
    # Minúsculas
    text = text.lower()

    # Quitar acentos
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

    # Quitar caracteres raros
    text = re.sub(r"[^a-zA-Z0-9áéíóúñü\s]", " ", text)

    # Quitar espacios múltiples
    text = re.sub(r"\s+", " ", text)

    return text.strip()
