import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Dataset
KAGGLE_DATASET = "mnassrib/telecom-churn-datasets"
RAW_CSV_NAME = "churn-bigml-80.csv"
CLEAN_CSV_NAME = "churn_clean.csv"

TARGET = "Churn"
CATEGORICAL_FEATURES = ["Area code"]
BINARY_FEATURES = ["International plan", "Voice mail plan"]
# Cargos con correlación 1.00 respecto a sus minutos (redundantes)
REDUNDANT_COLUMNS = [
    "Total day charge",
    "Total eve charge",
    "Total night charge",
    "Total intl charge",
]
# Alta cardinalidad: se excluye de las características
EXCLUDED_FEATURES = ["State"]

# Particiones estratificadas 70/15/15
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# MLflow (se pueden sobreescribir desde .env)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5002")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "Churn_Test_Telco_v1")

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
