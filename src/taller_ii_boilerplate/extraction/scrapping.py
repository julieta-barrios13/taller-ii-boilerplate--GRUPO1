import  requests
from bs4 import BeautifulSoup
import pandas as pd
from collections import defaultdict

url = "https://www.trustpilot.com/review/www.airbnb.com"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
                  " Chrome/113.0.0.0 Safari/537.36"
}

response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")

reviews = soup.find_all("article")  # Clase típica en Trustpilot, toda una reseña está dentro de un tag HTML article

review_dict = defaultdict(list)

for i, review in enumerate(reviews[:10]):
    review_text = review.find("p")
    rating = review.find("img").get("alt")
    review_dict['Review_text'].append(review_text.text.strip())
    review_dict['Rating'].append(rating)

review_data = pd.DataFrame(review_dict)
review_data.to_csv("/home/ana/Documents/UNSTA/Taller II/taller-ii-boilerplate/data/raw")
