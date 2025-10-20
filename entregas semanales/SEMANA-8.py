# Comparación entre GridSearchCV y RandomizedSearchCV para optimizar el modelo SVM + TF-IDF ("El mejor")

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from scipy.stats import uniform
import time

# Cargo y preparo dataset
df = pd.read_csv("entregas semanales/comentarios_preprocesados.csv")
text_col = "Review_clean" if "Review_clean" in df.columns else "Comment"

if "Sentiment" not in df.columns:
    def map_sentiment(score):
        if score <= 2: return "NEGATIVO"
        elif score == 3: return "NEUTRAL"
        else: return "POSITIVO"
    df["Sentiment"] = df["Score"].apply(map_sentiment)

X = df[text_col].astype(str)
y = df["Sentiment"].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Pipeline base
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('svm', LinearSVC(class_weight='balanced', random_state=42))
])

# Grillas de parámetros
param_grid = {
    'tfidf__max_features': [5000, 10000, 20000],
    'tfidf__ngram_range': [(1,1), (1,2)],
    'svm__C': [0.1, 0.5, 1, 2, 5],
    'svm__loss': ['hinge', 'squared_hinge']
}

param_distributions = {
    'tfidf__max_features': [3000, 5000, 10000, 20000],
    'tfidf__ngram_range': [(1,1), (1,2)],
    'svm__C': uniform(0.1, 5.0),
    'svm__loss': ['hinge', 'squared_hinge']
}

# Entreno con grid search
print("\n=== OPTIMIZACIÓN CON GRID SEARCH ===")
start = time.time()
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=2)
grid.fit(X_train, y_train)
grid_time = time.time() - start

# Evaluo grid search
y_pred_grid = grid.best_estimator_.predict(X_test)
print("\n>> Mejor configuración (GRID):", grid.best_params_)
print(">> Mejor F1 macro (CV=5):", grid.best_score_)
print(">> Tiempo de ejecución: %.2f s" % grid_time)
print(classification_report(y_test, y_pred_grid, digits=3))

# Entreno con random search
print("\n=== OPTIMIZACIÓN CON RANDOM SEARCH ===")
start = time.time()
random = RandomizedSearchCV(
    pipeline, param_distributions, n_iter=10, cv=5, scoring='f1_macro', n_jobs=-1, verbose=2, random_state=42
)
random.fit(X_train, y_train)
random_time = time.time() - start

# Evaluo random search
y_pred_random = random.best_estimator_.predict(X_test)
print("\n>> Mejor configuración (RANDOM):", random.best_params_)
print(">> Mejor F1 macro (CV=5):", random.best_score_)
print(">> Tiempo de ejecución: %.2f s" % random_time)
print(classification_report(y_test, y_pred_random, digits=3))

# Comparo finalmente ambos metodos
data = {
    "Método": ["GridSearchCV", "RandomizedSearchCV"],
    "F1_macro_CV": [grid.best_score_, random.best_score_],
    "Tiempo (s)": [grid_time, random_time],
}
resumen = pd.DataFrame(data)
print("\n=== COMPARACIÓN FINAL ===")
print(resumen)