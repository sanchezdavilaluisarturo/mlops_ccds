from dataclasses import dataclass, field
from pathlib import Path

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
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TEST_SIZE,
    VAL_SIZE,
)
from mlops_ccds.features import Splits, build_preprocessor, load_splits
from mlops_ccds.plots import plot_diagnostics, save_figure

app = typer.Typer()


@dataclass
class ModelSpec:
    run_name: str
    classifier: ClassifierMixin
    thresholds: np.ndarray
    param_grid: dict = field(default_factory=dict)


def get_model_spec(model: str, quick: bool = False) -> ModelSpec:
    if model == "baseline":
        return ModelSpec(
            run_name="Baseline_LogisticRegression",
            classifier=LogisticRegression(
                class_weight="balanced", random_state=RANDOM_STATE, max_iter=1000
            ),
            thresholds=np.linspace(0.01, 0.99, 100),
        )
    if model == "hgb":
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
            run_name="HistGradientBoosting_FineTuned",
            classifier=HistGradientBoostingClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                early_stopping=True,
                n_iter_no_change=15,
                tol=1e-4,
                scoring="roc_auc",
            ),
            thresholds=np.linspace(0.1, 0.9, 81),
            param_grid=grid,
        )
    raise typer.BadParameter(f"Modelo desconocido: {model!r}. Usa 'baseline' o 'hgb'.")


def build_pipeline(preprocessor: ColumnTransformer, classifier: ClassifierMixin) -> Pipeline:
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def fit_model(pipeline: Pipeline, splits: Splits, param_grid: dict):
    """Ajusta el pipeline. Con rejilla usa GridSearchCV (5 pliegues) optimizando ROC-AUC."""
    if not param_grid:
        return pipeline.fit(splits.X_train, splits.y_train), {}, None

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
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


def write_notes(run_name, best_params, threshold, metrics, reports_dir: Path) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"run_notes_{run_name}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Entrenamiento {run_name} finalizado.\n")
        if best_params:
            f.write(f"Mejores parámetros: {best_params}\n")
        f.write(f"Umbral óptimo en validación: {threshold:.4f}\n")
        f.write(f"Test ROC-AUC: {metrics['test_roc_auc']:.4f}\n")
        f.write(f"Test PR-AUC: {metrics['test_pr_auc']:.4f}\n")
        f.write(f"Test F1: {metrics['test_f1_score']:.4f}\n")
    return path


def log_run(
    run_name: str,
    spec: ModelSpec,
    pipeline: Pipeline,
    splits: Splits,
    best_params: dict,
    cv_score: float | None,
    threshold: float,
    val_f1: float,
    metrics: dict,
    plot_path: Path,
    notes_path: Path,
) -> str:
    """Registra params, métricas, gráficas, notas y el modelo en el servidor MLflow."""
    params = {
        "model_type": type(spec.classifier).__name__,
        "random_state": RANDOM_STATE,
        "train_split_pct": round(1 - TEST_SIZE - VAL_SIZE, 2),
        "val_split_pct": VAL_SIZE,
        "test_split_pct": TEST_SIZE,
        "optimal_threshold": round(threshold, 4),
    }
    if best_params:
        params.update(best_params)
    else:
        classifier_params = spec.classifier.get_params()
        params.update({k: classifier_params[k] for k in ("class_weight", "max_iter")})

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
    model: str = "baseline",
    run_name: str | None = None,
    quick: bool = False,
    data_dir: Path = PROCESSED_DATA_DIR,
    log_to_mlflow: bool = True,
):
    """Entrena un modelo (baseline | hgb), lo evalúa en Test y registra el run en MLflow."""
    spec = get_model_spec(model, quick)
    run_name = run_name or spec.run_name

    splits = load_splits(data_dir)
    pipeline = build_pipeline(build_preprocessor(splits.X_train), spec.classifier)

    logger.info(f"[{run_name}] Entrenando...")
    pipeline, best_params, cv_score = fit_model(pipeline, splits, spec.param_grid)
    if best_params:
        logger.info(f"Mejores parámetros: {best_params} | ROC-AUC CV: {cv_score:.4f}")

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
    notes_path = write_notes(run_name, best_params, threshold, metrics, REPORTS_DIR)

    if not log_to_mlflow:
        logger.warning("--no-log-to-mlflow: el run no se registró en MLflow.")
        return
    run_id = log_run(
        run_name, spec, pipeline, splits, best_params, cv_score,
        threshold, val_f1, metrics, plot_path, notes_path,
    )  # fmt: skip
    logger.success(f"RUN ID registrado en MLflow: {run_id}")


if __name__ == "__main__":
    app()
