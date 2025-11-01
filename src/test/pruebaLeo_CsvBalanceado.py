import pandas as pd
from sklearn.utils import resample

# Leer el CSV
df = pd.read_csv("entregas semanales/comentarios_preprocesados.csv")

# Crear columna 'Sentiment'
def map_sentiment(score):
    if score in [1, 2]:
        return "negativo"
    elif score == 3:
        return "neutro"
    else:
        return "positivo"

df["Sentiment"] = df["Score"].apply(map_sentiment)

print("Distribución original:")
print(df["Sentiment"].value_counts())

# Balanceo (oversampling)
df_neg = df[df["Sentiment"] == "negativo"]
df_neu = df[df["Sentiment"] == "neutro"]
df_pos = df[df["Sentiment"] == "positivo"]

max_count = df["Sentiment"].value_counts().max()

df_neu_up = resample(df_neu, replace=True, n_samples=max_count, random_state=42)
df_pos_up = resample(df_pos, replace=True, n_samples=max_count, random_state=42)

df_balanced = pd.concat([df_neg, df_neu_up, df_pos_up]).sample(frac=1, random_state=42)

print("\nDistribución balanceada:")
print(df_balanced["Sentiment"].value_counts())

# Guardar copia aparte (opcional)
df_balanced.to_csv("entregas semanales/comentarios_balanceados.csv", index=False)
print("\n💾 Guardado como 'comentarios_balanceados.csv'")
