from dataclasses import dataclass
from pathlib import Path

from loguru import logger
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import typer

from mlops_ccds.config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    CLEAN_CSV_NAME,
    EXCLUDED_FEATURES,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
    VAL_SIZE,
)
from mlops_ccds.dataset import clean_data, load_raw

app = typer.Typer()

SPLIT_NAMES = ("train", "val", "test")


@dataclass
class Splits:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def _restore_types(df: pd.DataFrame) -> pd.DataFrame:
    """Un CSV pierde el tipo texto de las variables categóricas; se restaura al leer."""
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str)
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE,
) -> Splits:
    """Partición estratificada en tres vías: test ciego, luego val sobre el resto."""
    X = df.drop(columns=[TARGET] + EXCLUDED_FEATURES)
    y = df[TARGET]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
    )
    return Splits(X_train, X_val, X_test, y_train, y_val, y_test)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Escalado numérico, One-Hot para categóricas y paso directo para binarias."""
    numeric = [c for c in X.columns if c not in CATEGORICAL_FEATURES + BINARY_FEATURES]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
        ]
    )


def save_splits(splits: Splits, output_dir: Path = PROCESSED_DATA_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_NAMES:
        X, y = getattr(splits, f"X_{name}"), getattr(splits, f"y_{name}")
        X.assign(**{TARGET: y}).to_csv(output_dir / f"{name}.csv", index=False)


def load_splits(input_dir: Path = PROCESSED_DATA_DIR) -> Splits:
    parts = {}
    for name in SPLIT_NAMES:
        df = _restore_types(pd.read_csv(input_dir / f"{name}.csv"))
        parts[f"X_{name}"] = df.drop(columns=[TARGET])
        parts[f"y_{name}"] = df[TARGET]
    return Splits(**parts)


def ensure_splits(data_dir: Path = PROCESSED_DATA_DIR, seed: int = RANDOM_STATE) -> Splits:
    """Lee las particiones de data/processed; si no existen, las genera desde el CSV crudo."""
    if all((data_dir / f"{name}.csv").exists() for name in SPLIT_NAMES):
        return load_splits(data_dir)

    logger.info(f"No hay particiones en {data_dir}; se generan desde los datos crudos.")
    splits = split_data(clean_data(load_raw()), random_state=seed)
    save_splits(splits, data_dir)
    return splits


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / CLEAN_CSV_NAME,
    output_dir: Path = PROCESSED_DATA_DIR,
    seed: int = typer.Option(RANDOM_STATE, help="Semilla aleatoria de la partición."),
):
    logger.info(f"Generando particiones train/val/test (semilla {seed})...")
    if input_path.exists():
        df = _restore_types(pd.read_csv(input_path))
    else:
        logger.info(f"{input_path} no existe; se limpia el CSV crudo.")
        df = clean_data(load_raw())
    splits = split_data(df, random_state=seed)
    save_splits(splits, output_dir)
    for name in SPLIT_NAMES:
        y = getattr(splits, f"y_{name}")
        logger.info(f"{name:5}: {len(y)} muestras | tasa Churn: {y.mean():.4f}")
    logger.success(f"Particiones guardadas en {output_dir}")


if __name__ == "__main__":
    app()
