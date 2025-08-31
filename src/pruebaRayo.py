import requests
from bs4 import BeautifulSoup

url = 'https://es.trustpilot.com/review/www.airbnb.es'
respuesta = requests.get(url)
page = 1

articulos_completos = []  

while respuesta.status_code == 200:
    sopa = BeautifulSoup(respuesta.text, 'html.parser')
    articulos = sopa.find_all('div', attrs={'data-testid': 'service-review-card-v2'})
    for articulo in articulos:
        # puntaje
        puntaje_tag = articulo.find('img', class_='CDS_StarRating_starRating__614d2e')
        puntaje = puntaje_tag['alt'] if puntaje_tag else 'Sin puntaje'
        # título
        titulo_tag = articulo.find('h2', class_='CDS_Typography_appearance-default__dd9b51 CDS_Typography_prettyStyle__dd9b51 CDS_Typography_heading-xs__dd9b51')
        titulo = titulo_tag.get_text() if titulo_tag else 'Sin título'
        # comentario
        comentario_tag = articulo.find('p', class_='CDS_Typography_appearance-default__dd9b51 CDS_Typography_prettyStyle__dd9b51 CDS_Typography_body-l__dd9b51')
        comentario = comentario_tag.get_text() if comentario_tag else 'Sin comentario'

        articulos_completos.append((titulo, comentario, puntaje))
    page += 1
    url = f'https://es.trustpilot.com/review/www.airbnb.es?page={page}'
    respuesta = requests.get(url)

# Guardar los comentarios
if articulos_completos:
    with open('comentarios.txt', 'w', encoding='utf-8') as f:
        for titulo, comentario, puntaje in articulos_completos:
            f.write(f'Título: {titulo}\nComentario: {comentario}\nPuntaje: {puntaje}\n\n')


