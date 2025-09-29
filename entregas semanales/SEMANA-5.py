import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

df = pd.read_csv("entregas semanales/comentarios_preprocesados.csv")
print("Tamaño del dataset:", df.shape)

# BoW
bow_vectorizer = CountVectorizer(max_features=1000)  # limitar a 1000 palabras más frecuentes
X_bow = bow_vectorizer.fit_transform(df["Review_clean"])

matriz_bow = pd.DataFrame(X_bow.toarray(), columns=bow_vectorizer.get_feature_names_out())
matriz_bow.to_csv("matriz_bow.csv", index=False, encoding="utf-8-sig")

print("\n=== Bag of Words ===")
print("Shape BoW:", X_bow.shape)
print("Ejemplo de vocabulario:", bow_vectorizer.get_feature_names_out()[:20])

# TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf_vectorizer.fit_transform(df["Review_clean"])

matriz_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
matriz_tfidf.to_csv("matriz_tfidf.csv", index=False, encoding="utf-8-sig")

print("\n=== TF-IDF ===")
print("Shape TF-IDF:", X_tfidf.shape)
print("Ejemplo de vocabulario:", tfidf_vectorizer.get_feature_names_out()[:20])

# Comparación simple en consola
print("\nComparación BoW vs TF-IDF en la primera reseña:")
print("BoW:", matriz_bow.iloc[0].head(10).to_dict())
print("TF-IDF:", matriz_tfidf.iloc[0].head(10).to_dict())
