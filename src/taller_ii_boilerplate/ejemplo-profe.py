import re
import pandas as pd
from bs4 import BeautifulSoup
import nltk
import emoji
import contractions
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
# Diccionario básico de chat words (se puede expandir)
chat_words = {
    "u": "you",
    "r": "are",
    "btw": "by the way",
    "idk": "i do not know",
}

def preprocess_text(text):
    if pd.isna(text):
        return ""

    # 1. Eliminar HTML
    text = BeautifulSoup(text, "html.parser").get_text()

    # 2. Expandir contracciones (don't -> do not)
    text = contractions.fix(text)

    # 4. Eliminar URLs
    text = re.sub(r"http\S+|www.\S+", "", text)

    # 5. Convertir emojis a texto
    text = emoji.demojize(text, delimiters=(" ", " "))

    # 6. Reemplazar chat words
    for word, full in chat_words.items():
        text = re.sub(r"\b" + word + r"\b", full, text)

    # 7. Eliminar números
    text = re.sub(r"\d+", "", text)

    # 9. Eliminar puntuación y caracteres especiales
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # 3. Pasar a minúsculas
    text = text.lower()

    # 10. Eliminar espacios extra
    text = re.sub(r"\s+", " ", text).strip()

    # 11. Tokenizar
    tokens = word_tokenize(text)

    # 12. Remover stopwords
    tokens = [w for w in tokens if w not in stop_words]

    # 13. Lematizar
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    return " ".join(tokens)


#Leemos el dataset de reseñas sin procesar
df = pd.read_csv("/Users/julietalujanbarrios/Desktop/TALLER II- Git/taller-ii-boilerplate--GRUPO1-1/entregas semanales/comentarios.csv")
print("Tamaño del dataset:", df.shape)
df.head()
# Aplicamos el preprocseamiento a la fila de reseña
df["review_clean"] = df["Comentario"].apply(preprocess_text)

print(df[["Comentario", "review_clean"]].head(10))