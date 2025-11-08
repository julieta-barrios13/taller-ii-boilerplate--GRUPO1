# cambios_chicho/remuestreo_final.py
import os
import re
import math
import pandas as pd
from sklearn.utils import resample

# === RUTAS ===
INPUT_PATH  = os.path.join("entregas semanales", "comentarios_preprocesados.csv")
OUTPUT_PATH = os.path.join("cambios_chicho", "comentarios_balanceados.csv")
RANDOM_STATE = 42

# === FUNCIONES AUXILIARES ===
def extract_score(row):
    """Devuelve el score numérico (1–5) desde las columnas Score o Qualification."""
    # 1) Si la columna Score existe y es válida
    if "Score" in row and pd.notna(row["Score"]):
        try:
            val = float(row["Score"])
            if 0 < val <= 5:
                return val
        except Exception:
            pass
    # 2) Si existe Qualification (e.g. "4 out of 5")
    q = row.get("Qualification", None)
    if isinstance(q, str):
        m = re.search(r"(\d+)", q)
        if m:
            try:
                val = float(m.group(1))
                if 0 < val <= 5:
                    return val
            except Exception:
                pass
    return math.nan


def score_to_sentiment(score):
    """Clasifica el score en positivo, neutral o negativo."""
    if pd.isna(score):
        return None
    if score >= 4:
        return "positive"
    if score <= 2:
        return "negative"
    return "neutral"  # cuando score == 3


# === REMUESTREO ===
def remuestrear(df):
    """Sobremuestrea positivas/neutrales y submuestrea negativas."""
    print("Distribución original:\n", df["Sentiment"].value_counts(), "\n")

    positive = df[df["Sentiment"] == "positive"]
    negative = df[df["Sentiment"] == "negative"]
    neutral  = df[df["Sentiment"] == "neutral"]

    # Cantidades base
    n_total = len(df)
    n_target = int(0.3 * n_total)  # 30% del total como referencia

    # Submuestreo de negativas (reduce)
    negative_down = resample(
        negative,
        replace=False,
        n_samples=min(len(negative), n_target),
        random_state=RANDOM_STATE
    )

    # Sobremuestreo de positivas (aumenta)
    positive_up = resample(
        positive,
        replace=True,
        n_samples=max(len(positive), n_target),
        random_state=RANDOM_STATE
    )

    # Sobremuestreo de neutrales (aumenta)
    neutral_up = resample(
        neutral,
        replace=True,
        n_samples=max(len(neutral), n_target),
        random_state=RANDOM_STATE
    )

    df_bal = pd.concat([negative_down, positive_up, neutral_up])
    df_bal = df_bal.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    print("Distribución balanceada:\n", df_bal["Sentiment"].value_counts(), "\n")
    return df_bal


# === MAIN ===
def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"No se encontró el archivo: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")

    # Inferir score y sentimiento
    df["Score_num"] = df.apply(extract_score, axis=1)
    df["Sentiment"] = df["Score_num"].apply(score_to_sentiment)

    # Filtrar los que no se pudieron clasificar
    df = df[pd.notna(df["Sentiment"])].copy()
    print(f"Total de reseñas clasificadas: {len(df)}")

    # Guardar dataset con etiquetas
    etiquetado_path = os.path.join("cambios_chicho", "comentarios_etiquetados.csv")
    os.makedirs(os.path.dirname(etiquetado_path), exist_ok=True)
    df.to_csv(etiquetado_path, index=False, encoding="utf-8-sig")
    print(f"Archivo etiquetado guardado en: {etiquetado_path}")

    # Aplicar remuestreo
    df_bal = remuestrear(df)

    # Guardar dataset balanceado
    df_bal.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"✅ Archivo balanceado guardado en: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
