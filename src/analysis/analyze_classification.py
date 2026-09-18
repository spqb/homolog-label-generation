#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

from common import METHOD_LABELS, REPO_ROOT, average_multiclass_roc, find_prediction_files, load_prediction_file, predictions_to_dataframe


DEFAULT_METHODS = ["onehot", "rbm", "esm2", "esmc", "ankh"]
DEFAULT_METHODS_RR = ["onehot", "rbm", "esm2", "esmc", "ankh", "esm2_supervised"]
ROC_COLORS = {
    "onehot": "#f96363", "rbm": "#ffb400", "esm2": "#00b170",
    "esmc": "#8e63ce", "ankh": "#a65628", "esm2_supervised": "#007acc",
}


def roc_method_style(model_id: str) -> dict:
    return {
        "color": ROC_COLORS.get(model_id),
        "linestyle": "--" if model_id in {"esm2", "esmc", "ankh", "esm2_supervised"} else "-",
    }


def short_method_label(model_id: str) -> str:
    return {"esm2": "ESM2", "esmc": "ESM C"}.get(model_id, METHOD_LABELS.get(model_id, model_id))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize repeated classification results across seeds.")
    parser.add_argument("--outputs-root", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--t1", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--train-sizes", type=int, nargs="+", default=[100, 500, 1000, 2000, 5000])
    parser.add_argument("--predictor", default="logreg", help="Prediction head stored in the HDF5 files.")
    parser.add_argument("--methods", nargs="+", default=None, help="Model ids to include.")
    parser.add_argument(
        "--plot-roc-labels", action="store_true",
        help="Also save a grid of one-vs-rest ROC curves, one panel per label, averaged across seeds.",
    )
    parser.add_argument(
        "--roc-train-size",
        type=int,
        default=100,
        help="Train size used for the averaged ROC plot.",
    )
    parser.add_argument(
        "--no-roc-error-band",
        action="store_true",
        help="Omit shaded across-seed variability bands from both ROC plots.",
    )
    parser.add_argument(
        "--use-latex",
        action="store_true",
        help="Use the same matplotlib LaTeX text configuration as the original notebook.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Figure directory (default: outputs/images/classification/<dataset>/t1_<t1>).")
    parser.add_argument("--tables-dir", type=Path, default=None, help="Metric table directory (default: outputs/analysis/classification/<dataset>/t1_<t1>/tables).")
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


def collect_metrics(dataset: str, t1: str, seeds: list[int], train_sizes: list[int], methods: list[str], predictor: str, outputs_root: Path = REPO_ROOT / "outputs") -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for seed in seeds:
        predictions_dir = outputs_root / dataset / f"t1_{t1}" / f"seed_{seed}" / "predictions"
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
    f1_labels = {
        "esm2": "ESM2",
        "esmc": "ESM C",
    }
    label_order = [f1_labels.get(method, METHOD_LABELS[method]) for method in method_order]
    metrics = metrics.copy()
    metrics["method"] = metrics["model_id"].map(
        lambda method: f1_labels.get(method, METHOD_LABELS[method])
    )
    fig, ax = plt.subplots(figsize=(6.25, 5.7), dpi=180)
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
    # ax.set_title(f"{dataset} - t1={t1}")
    ax.legend(title="Train size", loc="upper left", fontsize=13)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_roc_summary(dataset: str, t1: str, seeds: list[int], num_train_samples: int, methods: list[str], predictor: str, output_path: Path, outputs_root: Path = REPO_ROOT / "outputs", show_error_band: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(6, 5), dpi=180)

    for model_id in methods:
        curves = []
        auc_values = []
        for seed in seeds:
            path = outputs_root / dataset / f"t1_{t1}" / f"seed_{seed}" / "predictions" / f"test.embedding.{model_id}.ntrain_{num_train_samples}.predictions.h5"
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
        mean_auc = float(pd.Series(auc_values).mean())
        std_auc = float(pd.Series(auc_values).std(ddof=0))
        label = short_method_label(model_id)
        style = roc_method_style(model_id)

        ax.plot(curves[0][0], mean_tpr, **style, label=f"{label} (AUC = {mean_auc:.3f} ± {std_auc:.3f})")
        if show_error_band:
            std_tpr = pd.DataFrame([curve_tpr for _, curve_tpr in curves]).std(axis=0).to_numpy()
            ax.fill_between(curves[0][0], mean_tpr - std_tpr, mean_tpr + std_tpr, color=style["color"], alpha=0.2)

    ax.plot([0, 1], [0, 1], linestyle="-", linewidth=0.8, color="gray", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    # ax.set_title(f"{dataset} - t1={t1} - n={num_train_samples}")
    ax.legend(fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def per_label_roc(y_true: np.ndarray, probabilities: np.ndarray, classes: np.ndarray) -> dict:
    """Compute one-vs-rest curves using the probability-column class order."""
    probabilities = np.asarray(probabilities)
    if probabilities.ndim == 1:
        probabilities = probabilities[:, None]
    if len(classes) == 2 and probabilities.shape[1] == 1:
        probabilities = np.column_stack([1 - probabilities[:, 0], probabilities[:, 0]])
    if probabilities.shape != (len(y_true), len(classes)):
        raise ValueError("Probability dimensions do not match the sample count and class labels.")
    if not np.isin(y_true, classes).all():
        raise ValueError("Test labels are missing from the probability-column class labels.")
    grid = np.linspace(0, 1, 200)
    curves = {}
    for index, label in enumerate(classes):
        binary = np.asarray(y_true) == label
        if not binary.any() or binary.all():
            continue  # ROC requires both positive and negative examples.
        scores = probabilities[:, index]
        fpr, tpr, _ = roc_curve(binary, scores)
        curves[label] = (grid, np.interp(grid, fpr, tpr), roc_auc_score(binary, scores))
    return curves


def plot_roc_labels(dataset: str, t1: str, seeds: list[int], num_train_samples: int,
                    methods: list[str], predictor: str, output_path: Path,
                    outputs_root: Path = REPO_ROOT / "outputs", show_error_band: bool = True) -> None:
    by_label = {}
    for model_id in methods:
        for seed in seeds:
            path = outputs_root / dataset / f"t1_{t1}" / f"seed_{seed}" / "predictions" / f"test.embedding.{model_id}.ntrain_{num_train_samples}.predictions.h5"
            if not path.exists():
                continue
            loaded = load_prediction_file(path)
            key = f"{predictor}/labels_probs"
            if key not in loaded:
                continue
            for label in loaded["class_labels"]:
                by_label.setdefault(label, {})
            for label, curve in per_label_roc(loaded["labels_true"], loaded[key], loaded["class_labels"]).items():
                by_label[label].setdefault(model_id, []).append(curve)

    if not by_label:
        raise RuntimeError("No prediction files matched the per-label ROC grid configuration.")
    labels = sorted(by_label)
    ncols = min(3, len(labels))
    nrows = (len(labels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.3 * nrows),
                             sharex=True, sharey=True, squeeze=False, layout="constrained")
    for ax, label in zip(axes.flat, labels):
        for model_id in methods:
            curves = by_label[label].get(model_id, [])
            if not curves:
                continue
            tprs = np.stack([curve[1] for curve in curves])
            aucs = np.asarray([curve[2] for curve in curves])
            mean = tprs.mean(axis=0)
            line, = ax.plot(curves[0][0], mean, **roc_method_style(model_id),
                            label=f"{short_method_label(model_id)}: {aucs.mean():.3f} ± {aucs.std():.3f} (n={len(curves)})")
            if show_error_band:
                std = tprs.std(axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(mean)
                ax.fill_between(curves[0][0], np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1),
                                color=line.get_color(), alpha=0.2)
        ax.plot([0, 1], [0, 1], linestyle="-", linewidth=0.8, color="gray", label="Random")
        ax.set_title(str(label))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        if not by_label[label]:
            ax.text(0.5, 0.6, "No valid one-vs-rest splits", ha="center", transform=ax.transAxes)
        ax.legend(title="AUC (mean ± SD); n = seeds", loc="lower right", fontsize=8, title_fontsize=8)
    for ax in list(axes.flat)[len(labels):]:
        ax.set_visible(False)
    fig.supxlabel("False Positive Rate")
    fig.supylabel("True Positive Rate")
    fig.suptitle(f"{dataset} - t1={t1} - n={num_train_samples} - one-vs-rest ROC")
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
    output_dir = args.output_dir or args.outputs_root / "images" / "classification" / args.dataset / f"t1_{args.t1}"
    tables_dir = args.tables_dir or args.outputs_root / "analysis" / "classification" / args.dataset / f"t1_{args.t1}" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    methods = resolve_methods(args.dataset, args.methods)
    roc_train_size = args.roc_train_size

    metrics = collect_metrics(args.dataset, args.t1, args.seeds, args.train_sizes, methods, args.predictor, args.outputs_root)
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
        outputs_root=args.outputs_root,
        output_path=output_dir / f"{args.dataset}_t1_{args.t1}_roc_curves_ntrain_{roc_train_size}.pdf",
        show_error_band=not args.no_roc_error_band,
    )
    if args.plot_roc_labels:
        plot_roc_labels(
            args.dataset, args.t1, args.seeds, roc_train_size, methods, args.predictor,
            output_dir / f"{args.dataset}_t1_{args.t1}_roc_labels_ntrain_{roc_train_size}.pdf",
            outputs_root=args.outputs_root,
            show_error_band=not args.no_roc_error_band,
        )
    save_metric_tables(metrics, tables_dir, args.dataset, args.t1)


if __name__ == "__main__":
    main()
