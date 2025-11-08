import pandas as pd
from sklearn.utils import resample

# 1️⃣ Load the preprocessed CSV
df = pd.read_csv("entregas semanales/comentarios_preprocesados.csv")

# 2️⃣ Create sentiment column from 'Score'
def map_sentiment(score):
    if score in [1, 2]:
        return "negative"
    elif score == 3:
        return "neutral"
    else:
        return "positive"

df["Sentiment"] = df["Score"].apply(map_sentiment)

print("📊 Original class distribution:")
print(df["Sentiment"].value_counts())

# 3️⃣ Split by sentiment class
df_neg = df[df["Sentiment"] == "negative"]
df_neu = df[df["Sentiment"] == "neutral"]
df_pos = df[df["Sentiment"] == "positive"]

# 4️⃣ Find maximum class count
max_count = df["Sentiment"].value_counts().max()

# 5️⃣ Oversample minority classes to match the largest one
df_neu_up = resample(df_neu, replace=True, n_samples=max_count, random_state=42)
df_pos_up = resample(df_pos, replace=True, n_samples=max_count, random_state=42)

# 6️⃣ Combine and shuffle the balanced dataset
df_balanced = pd.concat([df_neg, df_neu_up, df_pos_up]).sample(frac=1, random_state=42)

print("\n Balanced class distribution:")
print(df_balanced["Sentiment"].value_counts())

# 7️⃣ Save the balanced version
df_balanced.to_csv("entregas semanales/comentarios_balanceados.csv", index=False)
print("\n💾 Saved as 'comentarios_balanceados.csv'")
