"""
Limpieza de los registros de libros extraídos.
"""

import glob
import os
import re

import pandas as pd

RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


def load_raw() -> pd.DataFrame:
    """Carga y concatena todos los CSV crudos con ``pd.concat``."""
    paths = sorted(glob.glob("data/raw*.csv"))
    if not paths:
        raise FileNotFoundError("No se encontraron archivos raw_*.csv en data/")
    frames = [pd.read_csv(path) for path in paths]
    # Concatenar fragmentos en un solo DataFrame; ignorar índices originales.
    return pd.concat(frames, ignore_index=True)


def _clean_price(value) -> float:
    """Elimina símbolos de moneda/comas y convierte a float."""
    if pd.isna(value):
        return value
    text = re.sub(r"[^0-9.]", "", str(value))
    return float(text) if text else float("nan")


def _clean_rating(value) -> int:
    """Mapea palabras de calificación a enteros del 1 al 5."""
    return RATING_MAP.get(str(value).strip(), 0)


def _clean_stock(value) -> int:
    """Extrae la cantidad disponible de 'In stock (N available)'."""
    if pd.isna(value):
        return 0
    match = re.search(r"(\d+)\s+available", str(value))
    if match:
        return int(match.group(1))
    # Alternativa: cualquier mención de stock cuenta como 1 unidad disponible.
    return 1 if "in stock" in str(value).lower() else 0


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica todos los pasos de limpieza y devuelve el DataFrame limpio."""
    # --- Inspección de duplicados / valores faltantes ---
    df = df.drop_duplicates(keep="first")
    df = df.dropna(subset=["title", "price_raw", "rating_raw"])
    df["availability_raw"] = df["availability_raw"].fillna("Out of stock")

    # --- Conversión de texto a numérico ---
    df["price"] = df["price_raw"].apply(_clean_price)
    df["rating"] = df["rating_raw"].apply(_clean_rating)
    df["stock"] = df["availability_raw"].apply(_clean_stock)

    # Eliminar columnas crudas intermedias una vez existen las versiones numéricas.
    df = df.drop(columns=["price_raw", "rating_raw", "availability_raw"])

    # --- Filtrado booleano multi-condición  ---
    # Conservar libros que sean (bien calificados Y en stock) O (baratos Y disponibles).
    # Los paréntesis agrupan estrictamente cada grupo de condiciones.
    mask = (
        ((df["rating"] >= 4) & (df["stock"] > 0))
        | ((df["price"] < 20.0) & (df["stock"] > 0))
    )
    df = df[mask].copy()

    # Validación: la calificación debe mantenerse entre 1 y 5.
    df = df[(df["rating"] >= 1) & (df["rating"] <= 5)]

    return df.reset_index(drop=True)


def main() -> None:
    os.makedirs("data", exist_ok=True)
    df = clean(load_raw())
    df.to_csv("data/clean_books.csv", index=False)
    print(f"Limpiados {len(df)} registros de libros -> data/clean_books.csv")


if __name__ == "__main__":
    main()
