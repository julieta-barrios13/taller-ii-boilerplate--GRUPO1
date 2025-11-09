import joblib
import gradio as gr
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
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =====================================================
# Load model and vectorizer
# =====================================================
vectorizer = joblib.load("entregas semanales/vectorizador_tfidf.pkl")
model = joblib.load("entregas semanales/modelo_mlp.pkl")


# =====================================================
# Prediction function
# =====================================================
def predict_sentiment(text):
    clean = preprocess_text(text)
    vector = vectorizer.transform([clean])
    pred = model.predict(vector)[0]
    proba = model.predict_proba(vector)[0]
    probabilities = {
        str(cls): round(float(p), 3) for cls, p in zip(model.classes_, proba)
    }
    return {"text": text, "prediction": pred, "probabilities": probabilities}


def interface(text):
    result = predict_sentiment(text)
    return result.get("prediction"), result.get("probabilities")


# =====================================================
# Custom CSS to match the screenshot design
# =====================================================
custom_css = """
body {
    background-color: #2b004f !important;
}

.gradio-container {
    background-color: #2b004f !important;
    color: #ff7b00 !important;
    font-family: 'Poppins', sans-serif;
}

h1, h2, h3, p, label {
    color: #ff7b00 !important;
    text-align: center;
}

#component-5 p {
    color: white !important;
}

h1 {
    color: #ff7b00 !important;
    font-weight: 700;
    letter-spacing: 1px;
}

textarea, input, .wrap.svelte-1clp57n {
    background-color: #9466b1 !important;
    border: 2px solid #ff7b00 !important;
    color: white !important;
}

textarea::placeholder {
    color: #d9bfe9 !important;
    font-weight: 600;
    text-align: center;
}

label.svelte-1ipelgc, .wrap.svelte-1ipelgc {
    color: #ff7b00 !important;
    font-weight: bold;
}

.output-markdown, .output-json, .label-wrap {
    background-color: #9466b1 !important;
    border: 2px solid #ff7b00 !important;
    color: #ff7b00 !important;
}

.output-markdown h2, .output-json h2 {
    color: #ff7b00 !important;
}

footer {
    display: none !important;
}
"""

# =====================================================
# Interface
# =====================================================
demo = gr.Interface(
    fn=interface,
    inputs=gr.Textbox(label="", lines=3, placeholder="Write something here...."),
    outputs=[
        gr.Label(label="Sentiment Prediction"),
        gr.JSON(label="Model Probabilities")
    ],
    title="SENTIMENT ANALYSIS",
    description="Write a text and the model will classify its sentiment as POSITIVE, NEGATIVE, or NEUTRAL.",
    css=custom_css
)

demo.launch()
