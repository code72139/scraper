# Análisis de Libros de E-commerce

Proyecto que extrae [Books to Scrape](http://books.toscrape.com/).

## Estructura
- `src/scraper.py` — web scraping (checkpoint crudo `data/raw_books.csv`)
- `src/cleaner.py` — limpieza y filtrado (`data/clean_books.csv`)
- `src/eda.py` — gráficos de EDA en `outputs/`
- `src/model.py` — modelo de clasificación (libros bien calificados)
- `main.py` — orquestador del pipeline

## Ejecución
```bash
pip install -r requirements.txt
python main.py
```

## Objetivo
Clasificación binaria: `high_rated = rating >= 4` estrellas, usando precio, stock y categoría.