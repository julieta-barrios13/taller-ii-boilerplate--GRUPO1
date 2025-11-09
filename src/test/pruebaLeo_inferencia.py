import joblib
import pandas as pd
import re
from bs4 import BeautifulSoup
import emoji

# =====================================================
# Simple English text preprocessing
# =====================================================
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = BeautifulSoup(text, "html.parser").get_text()
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # keep only English letters
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

# =====================================================
# Load trained model and vectorizer
# =====================================================
vectorizer = joblib.load("src/test/vectorizador_tfidf.pkl")
model = joblib.load("src/test/modelo_mlp.pkl")

# =====================================================
# Prediction function
# =====================================================
def predict_sentiment(text):
    clean = preprocess_text(text)
    vector = vectorizer.transform([clean])
    pred = model.predict(vector)[0]
    proba = model.predict_proba(vector)[0]
    probabilities = {
        cls: round(float(p), 3) for cls, p in zip(model.classes_, proba)
    }
    return {"text": text, "prediction": pred, "probabilities": probabilities}

# =====================================================
# Example usage
# =====================================================
if __name__ == "__main__":
    examples = [
    "The place was spotless and the host was very friendly.",
    "We waited two hours for check-in, totally unacceptable.",
    "Pretty average stay — not bad, but not impressive either.",
    "The view from the balcony was absolutely stunning!",
    "Customer service was terrible, they never answered my messages.",
    "Great location close to the subway, would stay again.",
    "It was fine, just like the pictures, nothing special though.",
    "The bathroom was small and smelled bad, but the bed was comfortable."
]

    for text in examples:
        result = predict_sentiment(text)
        print(f"\n📝 Text: {result['text']}")
        print(f"💬 Prediction: {result['prediction']}")
        print(f"📊 Probabilities: {result['probabilities']}")


