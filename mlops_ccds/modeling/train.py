from dataclasses import dataclass, field
import json
from pathlib import Path
import random
from typing import Annotated
import urllib.error
import urllib.request

from loguru import logger
import matplotlib.pyplot as plt
import mlflow
from mlflow.models.signature import infer_signature
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
import typer

from mlops_ccds.config import (
    FIGURES_DIR,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    RANDOM_STATE,
    REPORTS_DIR,
    TEST_SIZE,
    VAL_SIZE,
)
from mlops_ccds.dataset import clean_data, load_raw
from mlops_ccds.features import Splits, build_preprocessor, split_data
from mlops_ccds.plots import plot_diagnostics, save_figure

app = typer.Typer()


def set_seed(seed: int) -> None:
    """Fija las semillas globales de Python y NumPy.

    La reproducibilidad no depende solo de esto: la partición, el modelo y la validación
    cruzada reciben la misma semilla de forma explícita (random_state=seed).
    """
    random.seed(seed)
    np.random.seed(seed)


# Hiperparámetros que se registran en MLflow para cada modelo
LOGGED_HYPERPARAMS = {
    "baseline": ("C", "max_iter", "class_weight"),
    "hgb": (
        "learning_rate",
        "max_iter",
        "max_depth",
        "max_leaf_nodes",
        "min_samples_leaf",
        "l2_regularization",
        "class_weight",
    ),
}


@dataclass
class ModelSpec:
    model: str
    run_name: str
    classifier: ClassifierMixin
    thresholds: np.ndarray
    param_grid: dict = field(default_factory=dict)


def get_model_spec(
    model: str,
    seed: int = RANDOM_STATE,
    hyperparams: dict | None = None,
    tune: bool = False,
    quick: bool = False,
) -> ModelSpec:
    """Construye el clasificador con los hiperparámetros recibidos por línea de comandos.

    Con tune=True además se ajusta con GridSearchCV; las claves de la rejilla sustituyen a los
    valores fijos de esas mismas claves.
    """
    hp = hyperparams or {}

    if model == "baseline":
        classifier = LogisticRegression(
            class_weight="balanced",
            random_state=seed,
            C=hp.get("C", 1.0),
            max_iter=hp.get("max_iter", 1000),
        )
        return ModelSpec(
            model=model,
            run_name="Baseline_LogisticRegression",
            classifier=classifier,
            thresholds=np.linspace(0.01, 0.99, 100),
            param_grid={"classifier__C": [0.01, 0.1, 1.0, 10.0]} if tune else {},
        )

    if model == "hgb":
        classifier = HistGradientBoostingClassifier(
            class_weight="balanced",
            random_state=seed,
            early_stopping=True,
            n_iter_no_change=15,
            tol=1e-4,
            scoring="roc_auc",
            learning_rate=hp.get("learning_rate", 0.1),
            max_iter=hp.get("max_iter", 250),
            max_depth=hp.get("max_depth", 8),
            max_leaf_nodes=hp.get("max_leaf_nodes", 31),
            min_samples_leaf=hp.get("min_samples_leaf", 20),
            l2_regularization=hp.get("l2_regularization", 1.5),
        )
        grid = {}
        if tune:
            # --quick reduce la rejilla de 324 a 2 combinaciones, solo para pruebas
            grid = (
                {"classifier__learning_rate": [0.06, 0.10], "classifier__max_depth": [6]}
                if quick
                else {
                    "classifier__learning_rate": [0.03, 0.06, 0.10],
                    "classifier__max_iter": [250, 400],
                    "classifier__max_depth": [4, 6, 8],
                    "classifier__max_leaf_nodes": [15, 31],
                    "classifier__min_samples_leaf": [20, 35, 50],
                    "classifier__l2_regularization": [0.5, 1.5, 3.0],
                }
            )
        return ModelSpec(
            model=model,
            run_name="HistGradientBoosting_FineTuned" if tune else "HistGradientBoosting",
            classifier=classifier,
            thresholds=np.linspace(0.1, 0.9, 81),
            param_grid=grid,
        )

    raise typer.BadParameter(f"Modelo desconocido: {model!r}. Usa 'baseline' o 'hgb'.")


def build_pipeline(preprocessor: ColumnTransformer, classifier: ClassifierMixin) -> Pipeline:
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def fit_model(pipeline: Pipeline, splits: Splits, param_grid: dict, seed: int = RANDOM_STATE):
    """Ajusta el pipeline. Con rejilla usa GridSearchCV (5 pliegues) optimizando ROC-AUC."""
    if not param_grid:
        return pipeline.fit(splits.X_train, splits.y_train), {}, None

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    search = GridSearchCV(
        estimator=pipeline, param_grid=param_grid, scoring="roc_auc", cv=cv, n_jobs=-1, verbose=1
    )
    search.fit(splits.X_train, splits.y_train)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def select_threshold(pipeline: Pipeline, splits: Splits, thresholds: np.ndarray):
    """Umbral que maximiza F1 en Validación."""
    y_val_proba = pipeline.predict_proba(splits.X_val)[:, 1]
    f1_scores = [f1_score(splits.y_val, y_val_proba >= th, zero_division=0) for th in thresholds]
    best = int(np.argmax(f1_scores))
    return float(thresholds[best]), float(f1_scores[best])


def evaluate_model(pipeline: Pipeline, splits: Splits, threshold: float):
    """Métricas sobre el conjunto ciego de Test, con el umbral elegido en Validación."""
    y_proba = pipeline.predict_proba(splits.X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    metrics = {
        "test_roc_auc": float(roc_auc_score(splits.y_test, y_proba)),
        "test_pr_auc": float(average_precision_score(splits.y_test, y_proba)),
        "test_recall": float(recall_score(splits.y_test, y_pred, zero_division=0)),
        "test_precision": float(precision_score(splits.y_test, y_pred, zero_division=0)),
        "test_f1_score": float(f1_score(splits.y_test, y_pred, zero_division=0)),
    }
    return metrics, y_proba, y_pred


def write_notes(run_name, params, threshold, metrics, reports_dir: Path) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"run_notes_{run_name}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Entrenamiento {run_name} finalizado.\n")
        f.write(f"Hiperparámetros: {params}\n")
        f.write(f"Umbral óptimo en validación: {threshold:.4f}\n")
        f.write(f"Test ROC-AUC: {metrics['test_roc_auc']:.4f}\n")
        f.write(f"Test PR-AUC: {metrics['test_pr_auc']:.4f}\n")
        f.write(f"Test F1: {metrics['test_f1_score']:.4f}\n")
    return path


def write_metrics_json(run_name, seed, params, threshold, val_f1, metrics, reports_dir: Path):
    """Métricas con precisión completa: sirve para comparar dos ejecuciones bit a bit."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"metrics_{run_name}.json"
    payload = {
        "seed": seed,
        "hyperparameters": params,
        "optimal_threshold": threshold,
        "val_best_f1_score": val_f1,
        **metrics,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


def hyperparams_to_log(spec: ModelSpec, best_params: dict) -> dict:
    """Hiperparámetros efectivos del modelo: los de la rejilla ganadora o los fijos del CLI."""
    classifier_params = spec.classifier.get_params()
    params = {f"classifier__{k}": classifier_params[k] for k in LOGGED_HYPERPARAMS[spec.model]}
    params.update(best_params)
    return params


def mlflow_reachable(uri: str, timeout: float = 5.0) -> bool:
    """Comprueba que el servidor MLflow responde. Las URIs locales (sqlite, file) se aceptan."""
    if not uri.startswith(("http://", "https://")):
        return True
    try:
        urllib.request.urlopen(f"{uri.rstrip('/')}/health", timeout=timeout)
        return True
    except urllib.error.HTTPError:
        return True  # el servidor contestó, aunque no con 200
    except (OSError, ValueError):  # URLError y timeout son OSError
        return False


def log_run(
    run_name: str,
    seed: int,
    hyperparams: dict,
    pipeline: Pipeline,
    splits: Splits,
    cv_score: float | None,
    threshold: float,
    val_f1: float,
    metrics: dict,
    plot_path: Path,
    notes_path: Path,
    metrics_path: Path,
    model_type: str,
) -> str:
    """Registra params, métricas, gráficas, notas y el modelo en el servidor MLflow."""
    params = {
        "model_type": model_type,
        "random_state": seed,
        "train_split_pct": round(1 - TEST_SIZE - VAL_SIZE, 2),
        "val_split_pct": VAL_SIZE,
        "test_split_pct": TEST_SIZE,
        "optimal_threshold": round(threshold, 4),
        **hyperparams,
    }

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metric("val_best_f1_score", val_f1)
        if cv_score is not None:
            mlflow.log_metric("cv_best_train_roc_auc", cv_score)
        mlflow.log_metrics(metrics)

        mlflow.log_artifact(str(plot_path), artifact_path="evaluation_plots")
        mlflow.log_artifact(str(notes_path), artifact_path="metadata")
        mlflow.log_artifact(str(metrics_path), artifact_path="metadata")

        signature = infer_signature(splits.X_val, pipeline.predict(splits.X_val))
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            signature=signature,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
        return run.info.run_id


@app.command()
def main(
    model: Annotated[str, typer.Option(help="Modelo a entrenar: 'hgb' o 'baseline'.")] = "hgb",
    seed: Annotated[
        int, typer.Option(help="Semilla única: partición de datos, modelo y validación cruzada.")
    ] = RANDOM_STATE,
    learning_rate: Annotated[float, typer.Option(help="[hgb] Tasa de aprendizaje.")] = 0.1,
    max_depth: Annotated[int, typer.Option(help="[hgb] Profundidad máxima de cada árbol.")] = 8,
    max_leaf_nodes: Annotated[int, typer.Option(help="[hgb] Máximo de hojas por árbol.")] = 31,
    min_samples_leaf: Annotated[int, typer.Option(help="[hgb] Mínimo de muestras por hoja.")] = 20,
    l2_regularization: Annotated[float, typer.Option(help="[hgb] Regularización L2.")] = 1.5,
    max_iter: Annotated[
        int | None,
        typer.Option(
            help="[hgb, baseline] Iteraciones máximas (por defecto: 250 en hgb, 1000 en baseline)."
        ),
    ] = None,
    c: Annotated[
        float, typer.Option("--C", help="[baseline] Inverso de la regularización.")
    ] = 1.0,
    tune: Annotated[
        bool, typer.Option(help="Busca los hiperparámetros con GridSearchCV.")
    ] = False,
    quick: Annotated[
        bool, typer.Option(help="Con --tune, usa una rejilla mínima (pruebas).")
    ] = False,
    run_name: Annotated[str | None, typer.Option(help="Nombre del run en MLflow.")] = None,
    log_to_mlflow: Annotated[bool, typer.Option(help="Registra el run en MLflow.")] = True,
):
    """Entrena un modelo de churn, lo evalúa en Test y registra el run en MLflow.

    Sin argumentos entrena HistGradientBoosting con los hiperparámetros por defecto.
    """
    hyperparams = {
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "max_leaf_nodes": max_leaf_nodes,
        "min_samples_leaf": min_samples_leaf,
        "l2_regularization": l2_regularization,
        "C": c,
    }
    if max_iter is not None:
        hyperparams["max_iter"] = max_iter

    spec = get_model_spec(model, seed, hyperparams, tune, quick)
    run_name = run_name or spec.run_name

    if log_to_mlflow and not mlflow_reachable(MLFLOW_TRACKING_URI):
        logger.warning(
            f"MLflow no responde en {MLFLOW_TRACKING_URI}; se entrena sin registrar el run."
        )
        log_to_mlflow = False

    # La partición se recalcula siempre con la semilla recibida: el resultado depende solo
    # de los argumentos y del CSV crudo, no de archivos intermedios que pudieran estar viejos.
    set_seed(seed)
    splits = split_data(clean_data(load_raw()), random_state=seed)
    pipeline = build_pipeline(build_preprocessor(splits.X_train), spec.classifier)

    logger.info(f"[{run_name}] Entrenando (semilla {seed})...")
    pipeline, best_params, cv_score = fit_model(pipeline, splits, spec.param_grid, seed)
    if best_params:
        logger.info(f"Mejores parámetros: {best_params} | ROC-AUC CV: {cv_score:.4f}")
    logged_params = hyperparams_to_log(spec, best_params)
    logger.info(f"Hiperparámetros: {logged_params}")

    threshold, val_f1 = select_threshold(pipeline, splits, spec.thresholds)
    logger.info(f"Umbral óptimo en Val: {threshold:.4f} (F1 Val: {val_f1:.4f})")

    metrics, y_proba, y_pred = evaluate_model(pipeline, splits, threshold)
    print(
        classification_report(
            splits.y_test, y_pred, target_names=["No Churn", "Churn"], zero_division=0
        )
    )
    print(pd.DataFrame(list(metrics.items()), columns=["Métrica", "Puntaje"]).round(4))

    fig = plot_diagnostics(splits.y_test, y_proba, y_pred, threshold, metrics)
    plot_path = save_figure(fig, FIGURES_DIR / f"{run_name}_evaluation_plots.png")
    plt.close(fig)
    notes_path = write_notes(run_name, logged_params, threshold, metrics, REPORTS_DIR)
    metrics_path = write_metrics_json(
        run_name, seed, logged_params, threshold, val_f1, metrics, REPORTS_DIR
    )

    if not log_to_mlflow:
        logger.warning("El run no se registró en MLflow.")
        return
    run_id = log_run(
        run_name,
        seed,
        logged_params,
        pipeline,
        splits,
        cv_score,
        threshold,
        val_f1,
        metrics,
        plot_path,
        notes_path,
        metrics_path,
        type(spec.classifier).__name__,
    )
    logger.success(f"RUN ID registrado en MLflow: {run_id}")


if __name__ == "__main__":
    app()
