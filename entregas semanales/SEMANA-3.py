import re
import pandas as pd
from bs4 import BeautifulSoup
import nltk
import emoji
import contractions

# Descarga de recursos NLTK
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Stopwords y herramientas
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Diccionario de abreviaciones de chat
chat_words = {
    "u": "you",
    "r": "are",
    "btw": "by the way",
    "idk": "i do not know",
    "ur": "your",
    "thx": "thanks",
    "pls": "please",
}

# Diccionario de temas (ejemplo simple)
Topics = {
    "positive": ["good", "great", "love", "excellent", "amazing"],
    "negative": ["bad", "worst", "hate", "awful", "poor"],
    "service":  ["service", "staff", "support", "help"],
    "product":  ["product", "item", "quality", "price"]
}

def detect_topic(Tokens):
    """Detecta el tema más probable según palabras clave."""
    scores = {}
    for t, keywords in Topics.items():
        scores[t] = sum(token in keywords for token in Tokens)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "other"

def preprocess_text(text):
    if pd.isna(text):
        return "", [], [], []

    # 1. Eliminar HTML
    text = BeautifulSoup(text, "html.parser").get_text()

    # 2. Expandir contracciones (don't -> do not)
    text = contractions.fix(text)

    # 3. Eliminar URLs
    text = re.sub(r"http\S+|www.\S+", "", text)

    # 4. Convertir emojis a texto
    text = emoji.demojize(text, delimiters=(" ", " "))

    # 5. Reemplazar chat words
    for word, full in chat_words.items():
        text = re.sub(r"\b" + re.escape(word) + r"\b", full, text, flags=re.IGNORECASE)

    # 6. Reemplazar repeticiones de signos de puntuación (!!! -> !)
    text = re.sub(r"([!?.,])\1+", r"\1", text)

    # 7. Eliminar números
    text = re.sub(r"\d+", "", text)

    # 8. Pasar a minúsculas
    text = text.lower()

    # 9. Eliminar puntuación y caracteres especiales
    text = re.sub(r"[^a-z\s]", "", text)

    # 10. Eliminar espacios extra
    text = re.sub(r"\s+", " ", text).strip()

    # 11. Tokenizar
    Tokens = word_tokenize(text)

    # 12. Remover stopwords
    Tokens = [w for w in Tokens if w not in stop_words]

    # 13. Lemmatizar
    Lemmas = [lemmatizer.lemmatize(w, pos="v") for w in Tokens]

    # 14. Stemming
    Stems = [stemmer.stem(w) for w in Tokens]

    # Texto limpio final (unido)
    Clean_text = " ".join(Tokens)

    return Clean_text, Tokens, Lemmas, Stems


df = pd.read_csv("/Users/julietalujanbarrios/Desktop/TALLER II- Git/taller-ii-boilerplate--GRUPO1-1/entregas semanales/comentarios.csv")
print("Tamaño del dataset original:", df.shape)

# Aplicar procesamiento
df[["Review_clean", "Tokens", "Lemmas", "Stems"]] = df["Comentario"].apply(
    lambda x: pd.Series(preprocess_text(x))
)

# Tematización (basada en lemas para más precisión)
df["Topic"] = df["Lemmas"].apply(detect_topic)

# Mostrar primeras filas
print(df[["Comentario", "Review_clean", "Tokens", "Lemmas", "Stems", "Topic"]].head(10))

# Guardar CSV limpio
df.to_csv("comentarios_preprocesados.csv", index=False)
print("\nArchivo guardado como: comentarios_preprocesados.csv")
