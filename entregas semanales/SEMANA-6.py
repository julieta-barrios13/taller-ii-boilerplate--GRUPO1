import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

# 1) Cargar
df = pd.read_csv("entregas semanales/comentarios_preprocesados.csv")

# Mostrar columnas por si acaso
print("Columnas del CSV:", list(df.columns))

# Elegir columna de texto disponible
text_col = "Review_clean" if "Review_clean" in df.columns else "Comment"

# Si NO existe 'Sentiment', lo creamos desde 'Score'
if "Sentiment" in df.columns:
    y = df["Sentiment"].astype(str)
else:
    if "Score" not in df.columns:
        raise ValueError("No existen columnas 'Sentiment' ni 'Score' en el CSV.")
    # Asegurar que Score sea numérico
    score = pd.to_numeric(df["Score"], errors="coerce")

    # Filtrar filas válidas (score y texto presentes)
    mask = score.notna() & df[text_col].notna() & (df[text_col].astype(str).str.strip() != "")
    df = df.loc[mask].copy()
    score = score.loc[mask]

    # Mapear estrellas → sentimiento
    def map_sentiment(s):
        if s <= 2: return "neg"
        elif s == 3: return "neu"
        else: return "pos"
    df["Sentiment"] = score.apply(map_sentiment)
    y = df["Sentiment"].astype(str)

# Texto (relleno por seguridad)
X_text = df[text_col].astype(str).fillna("")

# 2) Split (estratificado si hay suficientes ejemplos por clase)
from collections import Counter as Cn
counts = Cn(y)
can_stratify = all(c >= 2 for c in counts.values())
X_train, X_test, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y if can_stratify else None
)
print("Distribución y_train:", Cn(y_train))
print("Distribución y_test :", Cn(y_test))

# 3) Vectorizadores (seguí con lo que ya tenías)
vectorizers = {
    "BoW": CountVectorizer(max_features=10000, ngram_range=(1,2)),
    "TFIDF": TfidfVectorizer(max_features=10000, ngram_range=(1,2)),
}


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import pandas as pd

vectorizers = {
    "BoW": CountVectorizer(max_features=20000, ngram_range=(1,2)),
    "TFIDF": TfidfVectorizer(max_features=20000, ngram_range=(1,2)),
}


# 4) Experimentos (modelos + HPs)

experiments = {
    "Naive Bayes": [
        MultinomialNB(alpha=1.0),
        MultinomialNB(alpha=0.1),
        MultinomialNB(alpha=0.01),
    ],
    "Logistic Regression": [
        LogisticRegression(max_iter=100,  class_weight="balanced", multi_class="ovr"),
        LogisticRegression(max_iter=1000, class_weight="balanced", multi_class="ovr"),
    ],
    "SVM": [
        LinearSVC(C=1.0, class_weight="balanced"),
        LinearSVC(C=2.0, class_weight="balanced"),
    ],
}


# 5) Baseline (clase mayoritaria)

from collections import Counter
maj = Counter(y_train).most_common(1)[0][0]
y_base = [maj] * len(y_test)
print("\n=== Baseline (clase mayoritaria) ===")
print(classification_report(y_test, y_base, digits=3))


# 6) Entrenar y evaluar todo

rows = []

for vec_name, vec in vectorizers.items():
    print(f"\n\n>>> Vectorizador: {vec_name}")
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    for family, models in experiments.items():
        for m in models:
            name = f"{m.__class__.__name__} ({vec_name})"
            print(f"\nEntrenando {name} ...")
            m.fit(Xtr, y_train)
            y_pred = m.predict(Xte)

            acc = accuracy_score(y_test, y_pred)
            f1m = f1_score(y_test, y_pred, average="macro")
            f1w = f1_score(y_test, y_pred, average="weighted")

            print(f"Accuracy: {acc:.3f} | Macro-F1: {f1m:.3f} | Weighted-F1: {f1w:.3f}")
            print("Reporte de clasificación:\n", classification_report(y_test, y_pred, digits=3))
            print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))

            rows.append({
                "Modelo": m.__class__.__name__,
                "Vectorización": vec_name,
                "Hiperparámetros": {k: v for k, v in m.get_params().items() if k in ("alpha","C","max_iter","class_weight","multi_class")},
                "Accuracy": acc,
                "Macro-F1": f1m,
                "Weighted-F1": f1w,
            })


# 7) Tabla comparativa final

resultados = pd.DataFrame(rows).sort_values(by=["Macro-F1","Weighted-F1"], ascending=False)
print("\n\n=== Comparación de modelos (ordenado por Macro-F1) ===")
print(resultados[["Modelo","Vectorización","Hiperparámetros","Accuracy","Macro-F1","Weighted-F1"]].to_string(index=False))
