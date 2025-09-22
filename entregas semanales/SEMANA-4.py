import pandas as pd
import numpy as np
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt


df = pd.read_csv("entregas semanales/comentarios_preprocesados.csv")
print("Tamaño del dataset original:", df.shape)

ax = sns.countplot(data=df, x="Score")
total = len(df)
for p in ax.patches:
    count = int(p.get_height())
    percentage = 100 * count / total
    ax.annotate(f'{count}\n({percentage:.1f}%)', 
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom')
plt.title("Distribución de Puntajes")
plt.xlabel("Puntaje")
plt.ylabel("Cantidad de Reseñas")
plt.show()


df['num_tokens'] = df['Tokens'].apply(lambda x: len(eval(x)) if isinstance(x, str) else len(x))
avg_tokens = df.groupby('Score')['num_tokens'].mean().reset_index()

plt.figure(figsize=(8, 6))
sns.barplot(data=avg_tokens, x='Score', y='num_tokens')
plt.title("Promedio de Tokens por Puntaje")
plt.xlabel("Puntaje")
plt.ylabel("Cantidad Promedio de Tokens")
plt.show()


# Unir todos los tokens en una sola lista
all_tokens = df['Tokens'].apply(lambda x: eval(x) if isinstance(x, str) else x)
flat_tokens = [token for tokens in all_tokens for token in tokens]

# Contar los tokens más frecuentes
token_counts = Counter(flat_tokens)
most_common_tokens = token_counts.most_common(10)

# Mostrar los 10 tokens más frecuentes
print("Tokens más frecuentes:")
for token, count in most_common_tokens:
    print(f"{token}: {count}")


# Nube de palabras para Score >= 4
tokens_high = df[df['Score'] >= 4]['Tokens'].apply(lambda x: eval(x) if isinstance(x, str) else x)
flat_tokens_high = [token for tokens in tokens_high for token in tokens]
text_high = ' '.join(flat_tokens_high)
wordcloud_high = WordCloud(width=800, height=400, background_color='white').generate(text_high)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_high, interpolation='bilinear')
plt.axis('off')
plt.title('Nube de Palabras - Score >= 4')
plt.show()

# Nube de palabras para Score <= 2
tokens_low = df[df['Score'] <= 2]['Tokens'].apply(lambda x: eval(x) if isinstance(x, str) else x)
flat_tokens_low = [token for tokens in tokens_low for token in tokens]
text_low = ' '.join(flat_tokens_low)
wordcloud_low = WordCloud(width=800, height=400, background_color='white').generate(text_low)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_low, interpolation='bilinear')
plt.axis('off')
plt.title('Nube de Palabras - Score <= 2')
plt.show()