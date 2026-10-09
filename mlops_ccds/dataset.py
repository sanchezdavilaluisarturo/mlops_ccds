from pathlib import Path
import shutil

from loguru import logger
import numpy as np
import pandas as pd
import typer

from mlops_ccds.config import (
    BINARY_FEATURES,
    CLEAN_CSV_NAME,
    KAGGLE_DATASET,
    PROCESSED_DATA_DIR,
    RAW_CSV_NAME,
    RAW_DATA_DIR,
    REDUNDANT_COLUMNS,
    TARGET,
)

app = typer.Typer()


def load_raw(csv_path: Path = RAW_DATA_DIR / RAW_CSV_NAME) -> pd.DataFrame:
    """Lee el CSV crudo. Si no está en data/raw, lo descarga de Kaggle y lo guarda ahí."""
    if not csv_path.exists():
        import kagglehub

        logger.info(f"{csv_path} no existe, descargando {KAGGLE_DATASET} de Kaggle...")
        downloaded = kagglehub.dataset_download(KAGGLE_DATASET)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(downloaded) / csv_path.name, csv_path)
    return pd.read_csv(csv_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Duplicados, tipos, mapeo binario, columnas redundantes e infinitos/nulos."""
    df = df.copy()

    n_dupes = int(df.duplicated().sum())
    if n_dupes > 0:
        df = df.drop_duplicates().reset_index(drop=True)
    logger.info(f"Duplicados eliminados: {n_dupes}")

    df["Area code"] = df["Area code"].astype(str)
    binary_map = {"Yes": 1, "No": 0, "yes": 1, "no": 0}
    for col in BINARY_FEATURES:
        df[col] = df[col].map(binary_map)
    df[TARGET] = df[TARGET].astype(int)

    df = df.drop(columns=REDUNDANT_COLUMNS, errors="ignore")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    logger.info(f"Tamaño tras la limpieza: {df.shape}")
    return df


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / RAW_CSV_NAME,
    output_path: Path = PROCESSED_DATA_DIR / CLEAN_CSV_NAME,
):
    logger.info("Procesando dataset...")
    df = clean_data(load_raw(input_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.success(f"Dataset limpio guardado en {output_path} ({df.shape[0]} filas).")


if __name__ == "__main__":
    app()
