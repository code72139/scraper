"""
Modelo de machine learning para predecir libros bien calificados.

"""

import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = "data/clean_books.csv"
MODEL_PATH = "outputs/model.pkl"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Carga el conjunto de datos limpio."""
    return pd.read_csv(path)


def build_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Crea features y un target binario (high_rated = rating >= 4)."""
    work = df.copy()
    work["high_rated"] = (work["rating"] >= 4).astype(int)
    features = work[["price", "stock", "category"]]
    target = work["high_rated"]
    return features, target


def make_preprocessor() -> ColumnTransformer:
    """ColumnTransformer: pasa numéricos, codifica categoría con one-hot."""
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", ["price", "stock"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["category"]),
        ]
    )


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """División train/test estratificada."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def train_baseline(X_train, y_train, X_test, y_test) -> tuple[Pipeline, float]:
    """Entrena un baseline RandomForest sin ajustar y devuelve su accuracy en test."""
    pipeline = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor()),
            ("clf", RandomForestClassifier(random_state=42)),
        ]
    )
    pipeline.fit(X_train, y_train)
    baseline_score = pipeline.score(X_test, y_test)
    return pipeline, baseline_score


def tune_model(pipeline: Pipeline, X_train, y_train, X_test, y_test):
    """Ejecuta GridSearchCV (cv=5) y devuelve el mejor estimador y su score ajustado."""
    param_grid = {
        "clf__n_estimators": [50, 100, 200],
        "clf__max_depth": [None, 5, 10],
        "clf__min_samples_split": [2, 5],
    }
    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    tuned_score = grid.best_estimator_.score(X_test, y_test)
    return grid, tuned_score


def main() -> None:
    os.makedirs("outputs", exist_ok=True)
    df = load_data()
    X, y = build_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    _, baseline_score = train_baseline(X_train, y_train, X_test, y_test)
    grid, tuned_score = tune_model(
        Pipeline(
            steps=[
                ("preprocessor", make_preprocessor()),
                ("clf", RandomForestClassifier(random_state=42)),
            ]
        ),
        X_train,
        y_train,
        X_test,
        y_test,
    )

    print("=== Resultados del Modelo ===")
    print(f"Baseline (RF por defecto) accuracy en test: {baseline_score:.4f}")
    print(f"Mejor accuracy en CV:                    {grid.best_score_:.4f}")
    print(f"Ajustado (GridSearchCV) accuracy en test: {tuned_score:.4f}")
    print(f"Mejores parámetros: {grid.best_params_}")

    joblib.dump(grid.best_estimator_, MODEL_PATH)
    print(f"Mejor modelo guardado -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
