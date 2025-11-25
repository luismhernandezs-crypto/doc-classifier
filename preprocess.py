import re
import unicodedata

def clean_text(text):

    text = text.lower()


    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )


    text = re.sub(r"[^a-zA-Z0-9áéíóúñü\s]", " ", text)


    text = re.sub(r"\s+", " ", text)

    return text.strip()
