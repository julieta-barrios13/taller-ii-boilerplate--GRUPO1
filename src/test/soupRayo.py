import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time

url = 'https://www.trustpilot.com/review/www.airbnb.com'
respuesta = requests.get(url)
page = 1

articulos_completos = []  

while respuesta.status_code == 200:
    sopa = BeautifulSoup(respuesta.text, 'html.parser')
    articulos = sopa.find_all('div', attrs={'data-testid': 'service-review-card-v2'})
    if not articulos:  
        break
    for articulo in articulos:
        # puntaje
        puntaje_tag = articulo.find('img', class_='CDS_StarRating_starRating__614d2e')
        puntaje = puntaje_tag['alt'] if puntaje_tag else 'Sin puntaje'
        # título
        titulo_tag = articulo.find('h2', class_='CDS_Typography_appearance-default__dd9b51 CDS_Typography_prettyStyle__dd9b51 CDS_Typography_heading-xs__dd9b51')
        titulo = titulo_tag.get_text(strip=True) if titulo_tag else 'Sin título'
        # comentario
        comentario_tag = articulo.find('p', class_='CDS_Typography_appearance-default__dd9b51 CDS_Typography_prettyStyle__dd9b51 CDS_Typography_body-l__dd9b51')
        comentario = comentario_tag.get_text(strip=True) if comentario_tag else 'Sin comentario'

        articulos_completos.append({
            "Title": titulo,
            "Comment": comentario,
            "Qualification": puntaje
        })
    
    page += 1
    url = f'https://www.trustpilot.com/review/www.airbnb.com?page={page}'
    time.sleep(3) # Espera 3 segundos antes de la siguiente consulta
    respuesta = requests.get(url)

# Convertir a DataFrame
df = pd.DataFrame(articulos_completos)

#Agregar índice con número de reseña (arranca en 1)
df.insert(0, "Id", range(1, len(df) + 1))

#Crear columna con el número de estrellas (extraer solo el número antes de "out")
df["Score"] = df["Qualification"].apply(
    lambda x: int(re.search(r"(\d+)", x).group(1)) if re.search(r"(\d+)", x) else None
)

# Mostrar en consola
print(df.head())

# Guardar en CSV
df.to_csv("comentarios.csv", index=False, encoding="utf-8-sig")
print("Archivo guardado como comentarios.csv")