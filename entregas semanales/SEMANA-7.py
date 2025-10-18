import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


# 1) Cargar CSV
df = pd.read_csv("entregas semanales/comentarios_preprocesados.csv")

print("Columnas del CSV:", list(df.columns))

# Columna de texto
text_col = "Review_clean" if "Review_clean" in df.columns else "Comment"

# Crear o usar 'Sentiment'
if "Sentiment" in df.columns:
    y = df["Sentiment"].astype(str)
else:
    if "Score" not in df.columns:
        raise ValueError("No existen columnas 'Sentiment' ni 'Score' en el CSV.")

    score = pd.to_numeric(df["Score"], errors="coerce")
    mask = score.notna() & df[text_col].notna() & (df[text_col].astype(str).str.strip() != "")
    df = df.loc[mask].copy()
    score = score.loc[mask]

    def map_sentiment(s):
        if s <= 2: return "NEGATIVO"
        elif s == 3: return "NEUTRAL"
        else: return "POSITIVO"
    df["Sentiment"] = score.apply(map_sentiment)
    y = df["Sentiment"].astype(str)

X_text = df[text_col].astype(str).fillna("")

# 2) Split
counts = Counter(y)
can_stratify = all(c >= 2 for c in counts.values())
X_train, X_test, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y if can_stratify else None
)
print("Distribución y_train:", Counter(y_train))
print("Distribución y_test :", Counter(y_test))

# === NUEVO: Codificar etiquetas ===
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# 3) Vectorizadores
vectorizers = {
    "BoW": CountVectorizer(max_features=20000, ngram_range=(1,2)),
    "TFIDF": TfidfVectorizer(max_features=20000, ngram_range=(1,2)),
}

# 4) Modelos y sus hiperparámetros
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
    "Random Forest": [
        RandomForestClassifier(n_estimators=100, max_depth=None, class_weight="balanced", random_state=42),
        RandomForestClassifier(n_estimators=300, max_depth=20, class_weight="balanced", random_state=42),
    ],
    "MLP (Red Neuronal)": [
        MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42),
        MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', max_iter=500, random_state=42),
    ],
    "XGBoost": [
        XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softmax",
            num_class=len(set(y)),
            random_state=42,
            use_label_encoder=False,
            eval_metric="mlogloss"
        ),
        XGBClassifier(
            n_estimators=300,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softmax",
            num_class=len(set(y)),
            random_state=42,
            use_label_encoder=False,
            eval_metric="mlogloss"
        ),
    ]
}

# 5) Baseline
maj = Counter(y_train).most_common(1)[0][0]
y_base = [maj] * len(y_test)
print("\n=== Baseline (clase mayoritaria) ===")
print(classification_report(y_test, y_base, digits=3))

# 6) Entrenamiento y evaluación
rows = []

for vec_name, vec in vectorizers.items():
    print(f"\n\n>>> Vectorizador: {vec_name}")
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    for family, models in experiments.items():
        for m in models:
            name = f"{m.__class__.__name__} ({vec_name})"
            print(f"\nEntrenando {name} ...")

            # Entrenar con etiquetas codificadas
            m.fit(Xtr, y_train_enc)

            # Predecir y decodificar etiquetas
            y_pred_enc = m.predict(Xte)
            y_pred = le.inverse_transform(y_pred_enc)

            acc = accuracy_score(y_test, y_pred)
            f1m = f1_score(y_test, y_pred, average="macro")
            f1w = f1_score(y_test, y_pred, average="weighted")

            print(f"Accuracy: {acc:.3f} | Macro-F1: {f1m:.3f} | Weighted-F1: {f1w:.3f}")
            print("Reporte de clasificación:\n", classification_report(y_test, y_pred, digits=3))
            print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))

            rows.append({
                "Familia": family,
                "Modelo": m.__class__.__name__,
                "Vectorización": vec_name,
                "Accuracy": acc,
                "Macro-F1": f1m,
                "Weighted-F1": f1w,
            })

# 7) Resultados finales
resultados = pd.DataFrame(rows).sort_values(by=["Macro-F1","Weighted-F1"], ascending=False)
print("\n\n=== Comparación de modelos (ordenado por Macro-F1) ===")
print(resultados[["Familia","Modelo","Vectorización","Accuracy","Macro-F1","Weighted-F1"]].to_string(index=False))

# 8) Identificar el mejor modelo y justificar
mejor = resultados.iloc[0]

print("\n\n=== MEJOR MODELO ===")
print(f"Familia: {mejor['Familia']}")
print(f"Modelo: {mejor['Modelo']}")
print(f"Vectorización: {mejor['Vectorización']}")
print(f"Accuracy: {mejor['Accuracy']:.3f}")
print(f"Macro-F1: {mejor['Macro-F1']:.3f}")
print(f"Weighted-F1: {mejor['Weighted-F1']:.3f}")

if "Random Forest" in mejor["Familia"]:
    razon = ("Random Forest suele rendir bien porque combina múltiples árboles de decisión, "
             "reduciendo el sobreajuste y capturando interacciones no lineales entre las palabras.")
elif "MLP" in mejor["Familia"]:
    razon = ("La red neuronal (MLP) aprende representaciones complejas del texto y puede captar patrones "
             "no lineales entre los vectores de palabras, mejorando el rendimiento en textos con matices.")
elif "XGBoost" in mejor["Familia"]:
    razon = ("XGBoost destaca por su capacidad para optimizar el error de clasificación mediante boosting, "
             "combinando múltiples modelos débiles y ajustando pesos para enfocarse en los ejemplos más difíciles.")
elif "SVM" in mejor["Familia"]:
    razon = ("El SVM logra buenos resultados en texto porque maximiza los márgenes entre clases en un espacio "
             "de alta dimensión, algo ideal cuando se usan vectores TF-IDF o n-gramas.")
elif "Logistic Regression" in mejor["Familia"]:
    razon = ("La regresión logística ofrece un buen equilibrio entre interpretabilidad y rendimiento, "
             "especialmente cuando las clases están balanceadas y las características son linealmente separables.")
elif "Naive Bayes" in mejor["Familia"]:
    razon = ("Naive Bayes es eficiente y suele funcionar bien con texto porque asume independencia entre palabras, "
             "lo que simplifica el modelo y reduce el riesgo de sobreajuste en conjuntos grandes.")
else:
    razon = "Este modelo obtuvo el mejor puntaje global en F1, reflejando una buena capacidad de generalización."

print(f"\n Justificación: {razon}")
