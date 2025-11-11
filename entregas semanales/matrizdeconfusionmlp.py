import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib
import time
import matplotlib.pyplot as plt

# 1 Cargar dataset balanceado
df = pd.read_csv("entregas semanales/comentarios_balanceados.csv")

X = df["Review_clean"].astype(str)
y = df["Sentiment"].astype(str)

# 2️ Dividir train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3️ Vectorización TF-IDF
vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4️ Modelo base MLP
mlp = MLPClassifier(max_iter=300, random_state=42)

# 5️ Grilla de hiperparámetros reducida
param_grid = {
    "hidden_layer_sizes": [(64,), (128,), (64, 32)],
    "activation": ["relu", "tanh"],
    "solver": ["adam"],
    "learning_rate_init": [0.001, 0.01],
}

# 6️ Grid Search
print("\n=== OPTIMIZACIÓN MLP ===")
start = time.time()
grid = GridSearchCV(
    mlp,
    param_grid=param_grid,
    cv=3,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=2
)
grid.fit(X_train_vec, y_train)
grid_time = time.time() - start

print("\nMejores hiperparámetros encontrados:")
print(grid.best_params_)
print(f"Tiempo total: {grid_time/60:.1f} minutos")
print("F1 macro promedio:", round(grid.best_score_, 3))

# 7️ Evaluación final
best_mlp = grid.best_estimator_
y_pred = best_mlp.predict(X_test_vec)

print("\n=== RESULTADOS EN TEST ===")
print(classification_report(y_test, y_pred, digits=3))

# 🔹 7.1 Matriz de confusión
print("\n=== MATRIZ DE CONFUSIÓN ===")
cm = confusion_matrix(y_test, y_pred, labels=best_mlp.classes_)
print(pd.DataFrame(cm, index=best_mlp.classes_, columns=best_mlp.classes_))

# Mostrar la matriz gráficamente
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_mlp.classes_)
disp.plot(cmap='Blues', values_format='d')
plt.title("Matriz de Confusión - MLP Optimizado")
plt.show()


