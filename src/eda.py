"""
Análisis exploratorio de datos del conjunto de libros.

"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

OUTPUT_DIR = "outputs"


def _ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(path: str = "data/clean_books.csv") -> pd.DataFrame:
    """Carga el conjunto de datos limpio."""
    return pd.read_csv(path)


def plot_price_distribution(df: pd.DataFrame) -> None:
    """Histograma + KDE de los precios de los libros."""
    plt.figure(figsize=(8, 5))
    sns.histplot(df["price"], kde=True, bins=20, color="steelblue")
    plt.title("Distribución de Precios de Libros")
    plt.xlabel("Precio (£)")
    plt.ylabel("Cantidad")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "price_distribution.png"))
    plt.close()


def plot_avg_price_by_category(df: pd.DataFrame) -> None:
    """Gráfico de barras del precio promedio de las 15 categorías principales."""
    top = (
        df.groupby("category")["price"]
        .mean()
        .sort_values(ascending=False)
        .head(15)
    )
    plt.figure(figsize=(9, 6))
    sns.barplot(x=top.values, y=top.index, hue=top.index, palette="viridis", legend=False)
    plt.title("Precio Promedio por Categoría (Top 15)")
    plt.xlabel("Precio Promedio (£)")
    plt.ylabel("Categoría")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "avg_price_by_category.png"))
    plt.close()


def plot_price_by_rating(df: pd.DataFrame) -> None:
    """Boxplot del precio agrupado por calificación (relación feature-target)."""
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="rating", y="price", data=df, hue="rating", palette="Set2", legend=False)
    plt.title("Distribución de Precios por Calificación en Estrellas")
    plt.xlabel("Calificación (estrellas)")
    plt.ylabel("Precio (£)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "price_by_rating.png"))
    plt.close()


def plot_rating_counts(df: pd.DataFrame) -> None:
    """Gráfico de conteo de calificaciones en estrellas."""
    plt.figure(figsize=(7, 5))
    sns.countplot(x="rating", data=df, hue="rating", palette="muted", legend=False)
    plt.title("Cantidad de Libros por Calificación en Estrellas")
    plt.xlabel("Calificación (estrellas)")
    plt.ylabel("Cantidad")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "rating_counts.png"))
    plt.close()


def print_insights(df: pd.DataFrame) -> None:
    """Imprime insights estadísticos en consola."""
    print("=== Insights de EDA ===")
    print(f"Libros analizados: {len(df)}")
    print(f"Precio: media={df['price'].mean():.2f}, mediana={df['price'].median():.2f}, "
          f"mín={df['price'].min():.2f}, máx={df['price'].max():.2f}")
    print(f"Calificación más común: {df['rating'].mode().iloc[0]} estrellas")
    high_rated = (df["rating"] >= 4).mean() * 100
    print(f"Bien calificados (>=4 estrellas): {high_rated:.1f}%")
    top_cat = df.groupby("category")["price"].mean().idxmax()
    print(f"Categoría con mayor precio promedio: {top_cat}")
    corr = df["price"].corr(df["rating"])
    print(f"Correlación(precio, calificación): {corr:.3f}")
    print(f"Gráficos guardados en {OUTPUT_DIR}/")


def main() -> None:
    _ensure_output_dir()
    sns.set_style("whitegrid")
    df = load_data()
    plot_price_distribution(df)
    plot_avg_price_by_category(df)
    plot_price_by_rating(df)
    plot_rating_counts(df)
    print_insights(df)


if __name__ == "__main__":
    main()
