# cambios_chicho/entrenar_balanceado.py
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

# === 1) Cargar CSV balanceado ===
df = pd.read_csv("cambios_chicho/comentarios_balanceados.csv", encoding="utf-8-sig")

print("Columnas del CSV:", list(df.columns))
print("Cantidad total de reseñas:", len(df))

# === 2) Definir texto y etiquetas ===
text_col = "Review_clean" if "Review_clean" in df.columns else "Comment"
y = df["Sentiment"].astype(str)
X_text = df[text_col].astype(str).fillna("")

# === 3) Split ===
counts = Counter(y)
print("Distribución original:", counts)
can_stratify = all(c >= 2 for c in counts.values())

X_train, X_test, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y if can_stratify else None
)
print("Distribución y_train:", Counter(y_train))
print("Distribución y_test :", Counter(y_test))

# === 4) Codificar etiquetas ===
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# === 5) Vectorizadores ===
stopwords = []  # si querés, podés volver a pegar tus stopwords personalizadas
vectorizers = {
    "BoW": CountVectorizer(max_features=20000, ngram_range=(1,2), stop_words=stopwords),
    "TFIDF": TfidfVectorizer(max_features=20000, ngram_range=(1,2), stop_words=stopwords),
}

# === 6) Modelos ===
experiments = {
    "Naive Bayes": [MultinomialNB(alpha=1.0)],
    "Logistic Regression": [
        LogisticRegression(max_iter=1000, class_weight="balanced", multi_class="ovr")
    ],
    "SVM": [LinearSVC(C=1.0, class_weight="balanced")],
    "Random Forest": [
        RandomForestClassifier(n_estimators=300, max_depth=None, class_weight="balanced", random_state=42)
    ],
    "MLP (Red Neuronal)": [
        MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', max_iter=500, random_state=42)
    ],
    "XGBoost": [
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
        )
    ]
}

# === 7) Entrenamiento y evaluación ===
rows = []

for vec_name, vec in vectorizers.items():
    print(f"\n\n>>> Vectorizador: {vec_name}")
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    for family, models in experiments.items():
        for m in models:
            name = f"{m.__class__.__name__} ({vec_name})"
            print(f"\n🧠 Entrenando {name} ...")

            m.fit(Xtr, y_train_enc)
            y_pred_enc = m.predict(Xte)
            y_pred = le.inverse_transform(y_pred_enc)

            acc = accuracy_score(y_test, y_pred)
            f1m = f1_score(y_test, y_pred, average="macro")
            f1w = f1_score(y_test, y_pred, average="weighted")

            print(f"✅ Accuracy: {acc:.3f} | Macro-F1: {f1m:.3f} | Weighted-F1: {f1w:.3f}")
            print("📊 Reporte de clasificación:\n", classification_report(y_test, y_pred, digits=3))
            print("🔢 Matriz de confusión:\n", confusion_matrix(y_test, y_pred))

            rows.append({
                "Familia": family,
                "Modelo": m.__class__.__name__,
                "Vectorización": vec_name,
                "Accuracy": acc,
                "Macro-F1": f1m,
                "Weighted-F1": f1w,
            })

# === 8) Comparación de resultados ===
resultados = pd.DataFrame(rows).sort_values(by=["Macro-F1","Weighted-F1"], ascending=False)
print("\n\n=== 🏆 COMPARACIÓN DE MODELOS (ordenado por Macro-F1) ===")
print(resultados[["Familia","Modelo","Vectorización","Accuracy","Macro-F1","Weighted-F1"]].to_string(index=False))

mejor = resultados.iloc[0]
print("\n\n=== 🥇 MEJOR MODELO ===")
print(f"Familia: {mejor['Familia']}")
print(f"Modelo: {mejor['Modelo']}")
print(f"Vectorización: {mejor['Vectorización']}")
print(f"Accuracy: {mejor['Accuracy']:.3f}")
print(f"Macro-F1: {mejor['Macro-F1']:.3f}")
print(f"Weighted-F1: {mejor['Weighted-F1']:.3f}")

print("\n✅ Entrenamiento finalizado correctamente.")
