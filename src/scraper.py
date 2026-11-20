"""Web scraper para books.toscrape.com.

Recopila registros crudos de libros (title, category, price_raw, rating_raw,
availability_raw) en todas las categorías y listados paginados, y los persiste
en ``data/raw_books.csv``.
"""

import os

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "http://books.toscrape.com/"
MAX_PAGES = 200 

RATING_WORDS = {"One", "Two", "Three", "Four", "Five"}


def _soup(url: str) -> BeautifulSoup:
    """Obtiene ``url`` y devuelve un objeto BeautifulSoup parseado."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or "utf-8"
    return BeautifulSoup(response.text, "lxml")


def discover_categories() -> list[tuple[str, str]]:
    """Devuelve pares ``(nombre, url)`` de cada categoría del menú lateral."""
    soup = _soup(BASE_URL)
    categories: list[tuple[str, str]] = []
    for link in soup.select("div.side_categories ul li ul li a"):
        name = link.get_text(strip=True)
        href = link["href"]
        url = href if href.startswith("http") else BASE_URL + href
        categories.append((name, url))
    return categories


def _parse_article(article, category: str) -> dict:
    """Extrae los campos crudos de un ``<article class="product_pod">``."""
    title = article.select_one("h3 a")["title"]

    price_el = article.select_one("p.price_color")
    price_raw = price_el.get_text(strip=True) if price_el else None

    rating_raw = None
    rating_el = article.select_one("p.star-rating")
    if rating_el:
        for css_class in rating_el.get("class", []):
            if css_class in RATING_WORDS:
                rating_raw = css_class
                break

    availability_el = article.select_one("p.availability")
    availability_raw = (
        availability_el.get_text(strip=True) if availability_el else None
    )

    return {
        "title": title,
        "category": category,
        "price_raw": price_raw,
        "rating_raw": rating_raw,
        "availability_raw": availability_raw,
    }


def scrape_listing_page(url: str, category: str) -> tuple[list[dict], BeautifulSoup]:
    """Extrae una página de listado; devuelve registros y el soup parseado."""
    soup = _soup(url)
    records = []
    for article in soup.select("article.product_pod"):
        record = _parse_article(article, category)
        # Enriquecer disponibilidad con el conteo de stock de la página de detalle.
        href = article.select_one("h3 a")["href"]
        detail_url = requests.compat.urljoin(url, href)
        try:
            detail_soup = _soup(detail_url)
            availability_el = detail_soup.select_one("p.availability")
            if availability_el:
                record["availability_raw"] = availability_el.get_text(strip=True)
        except requests.RequestException:
            pass
        records.append(record)
    return records, soup


def scrape_category(name: str, start_url: str) -> list[dict]:
    """Recorre todas las páginas paginadas de una categoría."""
    records: list[dict] = []
    url: str | None = start_url
    page = 1
    while url is not None and page <= MAX_PAGES:
        page_records, soup = scrape_listing_page(url, name)
        records.extend(page_records)

        next_link = soup.select_one("li.next a")
        if next_link:
            url = requests.compat.urljoin(url, next_link["href"])
            page += 1
        else:
            url = None
    return records


def scrape_all() -> pd.DataFrame:
    """Extrae todas las categorías y devuelve un DataFrame combinado."""
    all_records: list[dict] = []
    for name, url in discover_categories():
        all_records.extend(scrape_category(name, url))
    return pd.DataFrame(all_records)


def main() -> None:
    """Punto de entrada de la Fase 1: extrae y guarda el checkpoint crudo."""
    os.makedirs("data", exist_ok=True)
    df = scrape_all()
    df.to_csv("data/raw_books.csv", index=False)
    print(f"Extraídos {len(df)} registros crudos de libros -> data/raw_books.csv")


if __name__ == "__main__":
    main()
