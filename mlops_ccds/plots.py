from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
import typer

from mlops_ccds.config import FIGURES_DIR, RAW_CSV_NAME, RAW_DATA_DIR, TARGET
from mlops_ccds.dataset import load_raw

app = typer.Typer()

PALETTE = ["#2b5c8f", "#d95f02"]

sns.set_theme(style="whitegrid", palette="muted")


def save_figure(fig: plt.Figure, path: Path, dpi: int = 300) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


# ---------------------------------------------------------------- EDA


def explore_data(data: pd.DataFrame) -> None:
    print(f"Dimensiones: {data.shape[0]} filas × {data.shape[1]} columnas\n")
    data.info()
    print("\n=== Primeros 5 registros ===")
    print(data.head().T)
    print("\n=== Estadísticas descriptivas (numéricas) ===")
    print(data.describe().T)
    print("\n=== Estadísticas descriptivas (categóricas) ===")
    print(data.describe(include=["object", "bool"]).T)
    print(f"\nRegistros duplicados: {data.duplicated().sum()}")


def churn_rate_by_plan(data: pd.DataFrame) -> None:
    for col in ["International plan", "Voice mail plan"]:
        print(f"Tasa de Churn por {col} (%):")
        print(pd.crosstab(data[col], data[TARGET], normalize="index") * 100, "\n")


def find_high_correlations(data: pd.DataFrame, threshold: float = 0.9) -> pd.Series:
    corr = data.select_dtypes(include=["number"]).corr()
    upper = np.triu(np.ones(corr.shape), k=1).astype(bool)
    pairs = corr.where(upper).stack()
    high = pairs[pairs > threshold].sort_values(ascending=False)

    print("=" * 50)
    if high.empty:
        print(f"No se encontraron pares con una correlación mayor a {threshold}.")
    else:
        print(f"Pares con correlación superior a {threshold} (multicolinealidad crítica):")
        for (v1, v2), value in high.items():
            print(f"  • {v1:25} y {v2:25} -> {value:.4f}")
    print("=" * 50)
    return high


def plot_target(data: pd.DataFrame) -> plt.Figure:
    counts = data[TARGET].value_counts().sort_index()
    print("Frecuencia absoluta:\n", counts)
    print("\nPorcentaje (%):\n", (counts / counts.sum() * 100).round(2))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(x=TARGET, data=data, ax=ax[0], palette=PALETTE, hue=TARGET, legend=False)
    ax[0].set_title("Distribución de Clientes (Churn vs Retención)", fontweight="bold")
    ax[0].set_xlabel("¿Canceló el servicio? (Churn)")
    ax[0].set_ylabel("Cantidad de clientes")

    ax[1].pie(
        counts,
        labels=["No Churn (False)", "Churn (True)"],
        autopct="%1.1f%%",
        startangle=90,
        colors=PALETTE,
        explode=(0, 0.08),
    )
    ax[1].set_title("Proporción de Churn", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_correlation_matrix(data: pd.DataFrame) -> plt.Figure:
    numeric = data.select_dtypes(include=[np.number]).drop(columns="Area code", errors="ignore")
    fig = plt.figure(figsize=(14, 10))
    sns.heatmap(
        numeric.corr(),
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Matriz de Correlación (Variables Numéricas)")
    fig.tight_layout()
    return fig


def plot_histograms(data: pd.DataFrame) -> plt.Figure:
    num_cols = data.select_dtypes(include=["number"]).columns.tolist()
    n_cols = 3
    n_rows = int(np.ceil(len(num_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
    axes = np.array(axes).reshape(-1)

    for ax, var in zip(axes, num_cols):
        sns.histplot(data[var].dropna(), kde=True, color="teal", alpha=0.4, ax=ax)
        ax.set_title(f"Distribución de {var}", fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Frecuencia")
    for ax in axes[len(num_cols) :]:
        fig.delaxes(ax)

    fig.suptitle(
        "Distribución de Consumo y Llamadas (Telecom)", fontsize=18, fontweight="bold", y=1.00
    )
    fig.tight_layout()
    return fig


def plot_plans_vs_churn(data: pd.DataFrame) -> plt.Figure:
    plan_cols = ["International plan", "Voice mail plan"]
    fig, axes = plt.subplots(1, len(plan_cols), figsize=(12, 4))
    for ax, col in zip(axes, plan_cols):
        sns.countplot(data=data, x=col, hue=TARGET, palette=PALETTE, ax=ax)
        ax.set_title(f"Relación entre {col} y Churn", fontweight="bold")
        ax.set_ylabel("Cantidad de Clientes")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


def plot_service_calls(data: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(11, 5))
    sns.countplot(data=data, x="Customer service calls", hue=TARGET, palette=PALETTE)
    plt.title(
        "Impacto de Llamadas al Servicio al Cliente en el Churn", fontsize=13, fontweight="bold"
    )
    plt.ylabel("Cantidad de Clientes")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


def plot_day_usage(data: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col, title in zip(
        axes,
        ["Total day minutes", "Total day charge"],
        ["Minutos de Día vs Churn", "Cargos de Día ($) vs Churn"],
    ):
        sns.boxplot(data=data, x=TARGET, y=col, palette=PALETTE, hue=TARGET, legend=False, ax=ax)
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


# ----------------------------------------------------- Evaluación del modelo


def plot_diagnostics(
    y_true: pd.Series,
    y_proba: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
    metrics: dict,
) -> plt.Figure:
    """Matriz de confusión, curva ROC y curva Precision-Recall sobre Test.

    Las curvas se dibujan con roc_curve / precision_recall_curve y no con los *Display
    de sklearn, que cambian de argumentos entre versiones.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.heatmap(
        confusion_matrix(y_true, y_pred),
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=axes[0],
        xticklabels=["Pred: No Churn", "Pred: Churn"],
        yticklabels=["Real: No Churn", "Real: Churn"],
    )
    axes[0].set_title(f"Matriz de Confusión (Umbral = {threshold:.2f})")

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    axes[1].plot(fpr, tpr, color="#1f77b4", label=f"Modelo (AUC = {metrics['test_roc_auc']:.2f})")
    axes[1].plot([0, 1], [0, 1], "k--", label="Clasificador aleatorio (AUC = 0.50)")
    axes[1].set_xlabel("Tasa de falsos positivos")
    axes[1].set_ylabel("Tasa de verdaderos positivos")
    axes[1].set_title("Curva ROC (Test)")
    axes[1].legend()

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    axes[2].plot(
        recall, precision, color="#ff7f0e", label=f"Modelo (AP = {metrics['test_pr_auc']:.2f})"
    )
    axes[2].axhline(
        y_true.mean(), color="k", linestyle="--", label=f"Baseline aleatorio ({y_true.mean():.2f})"
    )
    axes[2].set_xlabel("Recall")
    axes[2].set_ylabel("Precision")
    axes[2].set_title("Curva Precision-Recall (Test)")
    axes[2].legend()

    fig.tight_layout()
    return fig


@app.command()
def main(input_path: Path = RAW_DATA_DIR / RAW_CSV_NAME, output_dir: Path = FIGURES_DIR):
    """Genera las gráficas del EDA sobre el dataset crudo y las guarda en reports/figures."""
    plt.switch_backend("Agg")
    data = load_raw(input_path)
    explore_data(data)
    find_high_correlations(data)

    figures = {
        "eda_target.png": plot_target,
        "eda_correlation.png": plot_correlation_matrix,
        "eda_histograms.png": plot_histograms,
        "eda_plans_vs_churn.png": plot_plans_vs_churn,
        "eda_service_calls.png": plot_service_calls,
        "eda_day_usage.png": plot_day_usage,
    }
    for filename, plot in figures.items():
        fig = plot(data)
        save_figure(fig, output_dir / filename)
        plt.close(fig)
        logger.info(f"Guardada {output_dir / filename}")
    logger.success("Gráficas del EDA completas.")


if __name__ == "__main__":
    app()
