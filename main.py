"""Punto de entrada que orquesta todo el pipeline de ciencia de datos."""

import os

from src import cleaner, eda, model, scraper


def main() -> None:
    # Fase 1: scraping (se omite si el checkpoint existe)
    if not os.path.exists("data/raw_books.csv"):
        print("=== Fase 1: Scraping ===")
        scraper.main()
    else:
        print("=== Fase 1: checkpoint crudo encontrado, se omite scraping ===")

    # Fase 2: limpieza
    print("=== Fase 2: Limpieza ===")
    cleaner.main()

    # Fase 3: EDA
    print("=== Fase 3: EDA ===")
    eda.main()

    # Fase 4: modelo
    print("=== Fase 4: Modelado ===")
    model.main()

    print("=== Pipeline completo ===")


if __name__ == "__main__":
    main()
