import requests
import pandas as pd

url = "https://dummyjson.com/products"

response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    productos = data["products"]

    df = pd.DataFrame(productos)[["id","title", "description", "rating", "price", "category"]]

    # Renombro columnas
    df = df.rename(columns={
        "id": "ID",
        "title": "Nombre del producto",
        "description": "Descripción",
        "rating": "Puntaje",
        "price": "Precio",
        "category": "Categoría"
    })

    # muestro las primeras filas
    print(df.head())

 
    df.to_csv("productos_filtrados.csv", index=False, encoding="utf-8-sig")
    print("Datos guardados en productos_filtrados.csv")

else:
    print("Error al obtener los datos:", response.status_code)
