import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1️⃣ Cargar el CSV balanceado
df = pd.read_csv("entregas semanales/comentarios_balanceados.csv")

# 2️⃣ Variables
X = df["Review_clean"]
y = df["Sentiment"]

# 3️⃣ Vectorización TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_vect = vectorizer.fit_transform(X)

# 4️⃣ Split train/test
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42, stratify=y)

# 5️⃣ Modelo SVM
svm = LinearSVC(random_state=42)
svm.fit(X_train, y_train)

# 6️⃣ Evaluación
y_pred = svm.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("\nClassification report:")
print(classification_report(y_test, y_pred))

# ============================================
# 🔁 Comparar SVM con dataset original vs balanceado
# ============================================

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# --- 1️⃣ Dataset ORIGINAL ---
df_orig = pd.read_csv("entregas semanales/comentarios_preprocesados.csv")

# Reagrupar scores en Sentiment
def map_sentiment(score):
    if score in [1, 2]:
        return "negativo"
    elif score == 3:
        return "neutro"
    else:
        return "positivo"

df_orig["Sentiment"] = df_orig["Score"].apply(map_sentiment)

X_orig = df_orig["Review_clean"]
y_orig = df_orig["Sentiment"]

# --- 2️⃣ Dataset BALANCEADO ---
df_bal = pd.read_csv("entregas semanales/comentarios_balanceados.csv")
X_bal = df_bal["Review_clean"]
y_bal = df_bal["Sentiment"]

# --- 3️⃣ Vectorización (mismo vectorizador para ambos) ---
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))

X_orig_vect = vectorizer.fit_transform(X_orig)
X_bal_vect = vectorizer.transform(X_bal)

# --- 4️⃣ Entrenamiento y evaluación con dataset ORIGINAL ---
X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(X_orig_vect, y_orig, test_size=0.2, random_state=42, stratify=y_orig)
svm_o = LinearSVC(random_state=42)
svm_o.fit(X_train_o, y_train_o)
y_pred_o = svm_o.predict(X_test_o)

print("\n🧠 RESULTADOS SVM - DATASET ORIGINAL")
print("Accuracy:", round(accuracy_score(y_test_o, y_pred_o), 3))
print(classification_report(y_test_o, y_pred_o))

# --- 5️⃣ Entrenamiento y evaluación con dataset BALANCEADO ---
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_bal_vect, y_bal, test_size=0.2, random_state=42, stratify=y_bal)
svm_b = LinearSVC(random_state=42)
svm_b.fit(X_train_b, y_train_b)
y_pred_b = svm_b.predict(X_test_b)

print("\n⚖️ RESULTADOS SVM - DATASET BALANCEADO")
print("Accuracy:", round(accuracy_score(y_test_b, y_pred_b), 3))
print(classification_report(y_test_b, y_pred_b))
