#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import METHOD_LABELS, REPO_ROOT, average_multiclass_roc, find_prediction_files, load_prediction_file, predictions_to_dataframe


DEFAULT_METHODS = ["onehot", "rbm", "plm"]
DEFAULT_METHODS_RR = ["onehot", "rbm", "plm", "plm_supervised"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize repeated classification results across seeds.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--t1", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--train-sizes", type=int, nargs="+", default=[100, 500, 1000, 2000, 5000])
    parser.add_argument("--predictor", default="logreg", help="Prediction head stored in the HDF5 files.")
    parser.add_argument("--methods", nargs="+", default=None, help="Model ids to include.")
    parser.add_argument(
        "--roc-train-size",
        type=int,
        default=100,
        help="Train size used for the averaged ROC plot.",
    )
    parser.add_argument(
        "--use-latex",
        action="store_true",
        help="Use the same matplotlib LaTeX text configuration as the original notebook.",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "images" / "analysis")
    return parser.parse_args()


def resolve_methods(dataset: str, requested_methods: list[str] | None) -> list[str]:
    if requested_methods is not None:
        return requested_methods
    if dataset == "RR":
        return DEFAULT_METHODS_RR
    return DEFAULT_METHODS


def configure_matplotlib(use_latex: bool) -> None:
    if use_latex:
        plt.rc("text", usetex=True)
        plt.rc("font", size=14)


def collect_metrics(dataset: str, t1: str, seeds: list[int], train_sizes: list[int], methods: list[str], predictor: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for seed in seeds:
        predictions_dir = REPO_ROOT / "outputs" / dataset / f"t1_{t1}" / f"seed_{seed}" / "predictions"
        frame = predictions_to_dataframe(find_prediction_files(predictions_dir), predictor=predictor)
        if frame.empty:
            continue
        frame = frame[frame["model_id"].isin(methods) & frame["num_train_samples"].isin(train_sizes)].copy()
        frame["seed"] = seed
        rows.append(frame)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def plot_f1_summary(metrics: pd.DataFrame, dataset: str, t1: str, output_path: Path) -> None:
    method_order = [method for method in DEFAULT_METHODS_RR if method in metrics["model_id"].unique()]
    label_order = [METHOD_LABELS[method] for method in method_order]
    fig, ax = plt.subplots(figsize=(7, 5), dpi=180)
    sns.barplot(
        data=metrics,
        x="method",
        y="f1_score",
        hue="num_train_samples",
        order=label_order,
        ax=ax,
        palette="YlGnBu",
        edgecolor="black",
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Method")
    ax.set_ylabel("Macro F1")
    ax.set_title(f"{dataset} - t1={t1}")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_roc_summary(dataset: str, t1: str, seeds: list[int], num_train_samples: int, methods: list[str], predictor: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5), dpi=180)
    colors = {
        "onehot": "#f96363",
        "rbm": "#ffb400",
        "plm": "#00b170",
        "plm_supervised": "#007acc",
    }

    for model_id in methods:
        curves = []
        auc_values = []
        for seed in seeds:
            path = REPO_ROOT / "outputs" / dataset / f"t1_{t1}" / f"seed_{seed}" / "predictions" / f"test.embedding.{model_id}.ntrain_{num_train_samples}.predictions.h5"
            if not path.exists():
                continue
            loaded = load_prediction_file(path)
            probs_key = f"{predictor}/labels_probs"
            if probs_key not in loaded:
                continue
            fpr, tpr, auc = average_multiclass_roc(loaded["labels_true"], loaded[probs_key])
            curves.append((fpr, tpr))
            auc_values.append(auc)

        if not curves:
            continue

        mean_tpr = sum(curve_tpr for _, curve_tpr in curves) / len(curves)
        std_tpr = pd.DataFrame([curve_tpr for _, curve_tpr in curves]).std(axis=0).to_numpy()
        mean_auc = float(pd.Series(auc_values).mean())
        std_auc = float(pd.Series(auc_values).std(ddof=0))
        label = METHOD_LABELS.get(model_id, model_id)
        color = colors.get(model_id, None)

        ax.plot(curves[0][0], mean_tpr, color=color, label=f"{label} (AUC = {mean_auc:.3f} ± {std_auc:.3f})")
        ax.fill_between(curves[0][0], mean_tpr - std_tpr, mean_tpr + std_tpr, color=color, alpha=0.2)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{dataset} - t1={t1} - n={num_train_samples}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_metric_tables(metrics: pd.DataFrame, output_dir: Path, dataset: str, t1: str) -> None:
    metrics.to_csv(output_dir / f"{dataset}_t1_{t1}_classification_metrics_by_seed.csv", index=False)
    summary = (
        metrics.groupby(["method", "model_id", "num_train_samples"], as_index=False)
        .agg(
            f1_score_mean=("f1_score", "mean"),
            f1_score_std=("f1_score", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
        )
    )
    summary.to_csv(output_dir / f"{dataset}_t1_{t1}_classification_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    configure_matplotlib(args.use_latex)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = resolve_methods(args.dataset, args.methods)
    roc_train_size = args.roc_train_size

    metrics = collect_metrics(args.dataset, args.t1, args.seeds, args.train_sizes, methods, args.predictor)
    if metrics.empty:
        raise RuntimeError("No classification prediction files matched the requested configuration.")

    plot_f1_summary(metrics, args.dataset, args.t1, output_dir / f"{args.dataset}_t1_{args.t1}_f1_scores.pdf")
    plot_roc_summary(
        args.dataset,
        args.t1,
        args.seeds,
        num_train_samples=roc_train_size,
        methods=methods,
        predictor=args.predictor,
        output_path=output_dir / f"{args.dataset}_t1_{args.t1}_roc_curves_ntrain_{roc_train_size}.pdf",
    )
    save_metric_tables(metrics, output_dir, args.dataset, args.t1)


if __name__ == "__main__":
    main()
