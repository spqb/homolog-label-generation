#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

from adabmDCA.fasta import encode_sequence, get_tokens
from adabmDCA.stats import get_correlation_two_points, get_freq_single_point, get_freq_two_points

from common import REPO_ROOT, decode_array


DISPLAY_TO_ID = {
    "True": "true",
    "Train-only": "onlytrain",
    "One-hot": "onehot",
    "RBM": "rbm",
    "ESM2 650M": "esm2",
    "ESM C 600M": "esmc",
    "Ankh": "ankh",
}
ID_TO_DISPLAY = {value: key for key, value in DISPLAY_TO_ID.items()}
COLOR_BY_DISPLAY = {
    "True": "#0072b2",
    "Train-only": "#7a7a7a",
    "One-hot": "#f96363",
    "RBM": "#ffb400",
    "ESM2 650M": "#00b170",
    "ESM C 600M": "#8e63ce",
    "Ankh": "#a65628",
}
MARKER_BY_DISPLAY = {
    "True": "o",
    "Train-only": "X",
    "One-hot": "s",
    "RBM": "D",
    "ESM2 650M": "^",
    "ESM C 600M": "v",
    "Ankh": "P",
}
DEFAULT_GENERATION_METHODS = ["true", "onehot", "rbm", "ankh"]
FALLBACK_MODELS_ROOT = REPO_ROOT / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the main generated-sample analyses from sample_rbm.ipynb.")
    parser.add_argument("--dataset", default="RR")
    parser.add_argument("--t1", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    parser.add_argument("--kl-seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    parser.add_argument("--train-sizes", type=int, nargs="+", default=[100, 500, 1000, 2000, 5000])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(ID_TO_DISPLAY),
        default=DEFAULT_GENERATION_METHODS,
        help="Methods for quantitative generation and self-consistency analysis "
        "(default: true onehot rbm ankh). Train-only is reserved for the separate single-seed PCA plot.",
    )
    parser.add_argument("--outputs-root", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--models-root", type=Path, default=REPO_ROOT / "models")
    parser.add_argument("--samples-root", type=Path, default=None, help="Case-level generated-artifact root; defaults to the legacy models root.")
    parser.add_argument("--predictor", default="logreg")
    parser.add_argument("--roc-train-size", type=int, default=100)
    parser.add_argument("--pca-seed", type=int, default=1)
    parser.add_argument("--pca-train-size", type=int, default=100)
    parser.add_argument("--pearson-train-size", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=None, help="Figure directory (default: outputs/images/generation_natural/<dataset>/t1_<t1>).")
    parser.add_argument("--tables-dir", type=Path, default=None, help="Table directory (default: outputs/analysis/generation_natural/<dataset>/t1_<t1>/tables).")
    parser.add_argument("--plot-rr-pca", action="store_true", help="Render the integrated RR four-column PCA comparison and separate True figure.")
    parser.add_argument("--pca-only", action="store_true", help="Render only the RR PCA figures; implies --plot-rr-pca.")
    parser.add_argument("--max-scatter-points", type=int, default=3000)
    parser.add_argument("--use-latex", action="store_true")
    return parser.parse_args()


def configure_matplotlib(use_latex: bool) -> None:
    if use_latex:
        plt.rcParams.update({"text.usetex": True, "font.size": 14})


def natural_dataset_path(outputs_root: Path, dataset: str, t1: str, seed: int, train_size: int) -> Path:
    return outputs_root / dataset / f"t1_{t1}" / f"seed_{seed}" / "datasets_for_rbm" / f"rbm_dataset_ntrain_{train_size}.csv"


def sample_dir(models_root: Path, dataset: str, t1: str, seed: int, method_id: str, train_size: int) -> Path:
    name = "RBM_labels_all_true_full" if method_id == "true" else f"RBM_labels_{train_size}_{method_id}"
    generated = models_root / dataset / f"t1_{t1}" / f"seed_{seed}" / "generation" / name
    if generated.is_dir():
        return generated
    direct_legacy = models_root / dataset / f"t1_{t1}" / f"seed_{seed}" / name
    if direct_legacy.is_dir():
        return direct_legacy
    legacy = FALLBACK_MODELS_ROOT / dataset / f"t1_{t1}" / f"seed_{seed}" / name
    if legacy.is_dir():
        return legacy
    return generated


def sample_csv_path(models_root: Path, dataset: str, t1: str, seed: int, method_id: str, train_size: int) -> Path:
    """Samples conditioned on true_label for natural-distribution comparisons."""
    return sample_dir(models_root, dataset, t1, seed, method_id, train_size) / "samples.true_labels.csv"


def prediction_h5_path(models_root: Path, dataset: str, t1: str, seed: int, method_id: str, train_size: int) -> Path:
    suffix = "true" if method_id == "true" else method_id
    return sample_dir(models_root, dataset, t1, seed, method_id, train_size) / f"samples.self_consistency.predictions.{suffix}.h5"


def artifact_coverage(
    dataset: str, t1: str, seeds: list[int], train_sizes: list[int], methods: list[str], models_root: Path
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in seeds:
        for train_size in train_sizes:
            for method_id in methods:
                sample_path = sample_csv_path(models_root, dataset, t1, seed, method_id, train_size)
                prediction_path = prediction_h5_path(models_root, dataset, t1, seed, method_id, train_size)
                rows.append(
                    {
                        "seed": seed,
                        "num_train_samples": train_size,
                        "method_id": method_id,
                        "natural_samples_present": sample_path.is_file() and sample_path.stat().st_size > 0,
                        "predictions_expected": method_id != "onlytrain",
                        "predictions_present": (
                            method_id != "onlytrain"
                            and prediction_path.is_file()
                            and prediction_path.stat().st_size > 0
                        ),
                    }
                )
    return pd.DataFrame(rows)


def load_one_hot_flat(csv_path: Path, tokens: list[str], sequence_column: str = "sequence_align") -> tuple[np.ndarray, pd.DataFrame]:
    frame = pd.read_csv(csv_path)
    encoded = encode_sequence(frame[sequence_column].tolist(), tokens)
    one_hot = np.eye(len(tokens))[encoded]
    return one_hot.reshape(len(one_hot), -1), frame


def conditioning_labels(frame: pd.DataFrame, method_id: str, path: Path) -> np.ndarray:
    """Return prescribed true labels for natural-distribution comparisons."""
    if "true_label" not in frame.columns:
        raise ValueError(f"Natural-comparison samples must contain true_label: {path}")
    return frame["true_label"].astype(str).to_numpy()


def average_multiclass_roc(
    y_true: np.ndarray, y_probs: np.ndarray, class_labels: np.ndarray, npoints: int = 100
) -> tuple[np.ndarray, np.ndarray, float]:
    labels = np.asarray(class_labels)
    probs = np.asarray(y_probs)
    if probs.ndim == 1:
        probs = probs[:, None]
    if len(labels) == 2 and probs.shape[1] == 1:
        probs = np.column_stack([1.0 - probs[:, 0], probs[:, 0]])

    grid = np.linspace(0.0, 1.0, npoints)
    if probs.shape[1] != len(labels):
        raise ValueError(f"Expected {len(labels)} probability columns, found {probs.shape[1]}")
    mean_tpr = np.zeros_like(grid)
    auc_values = []
    for index, label in enumerate(labels):
        y_bin = (y_true == label).astype(int)
        if np.unique(y_bin).size < 2:
            continue
        fpr, tpr, _ = roc_curve(y_bin, probs[:, index])
        mean_tpr += np.interp(grid, fpr, tpr)
        auc_values.append(roc_auc_score(y_bin, probs[:, index]))
    if not auc_values:
        return grid, np.full_like(grid, np.nan), float("nan")
    mean_tpr /= len(auc_values)
    return grid, mean_tpr, float(np.mean(auc_values))


def histogram_divergences_by_label(
    data_pca: np.ndarray,
    labels: np.ndarray,
    gen_pca: np.ndarray,
    generated_labels: np.ndarray,
    method_id: str,
    num_train_samples: int,
    seed: int,
    bins: int = 40,
    eps: float = 1e-10,
) -> pd.DataFrame:
    all_points = np.vstack([data_pca, gen_pca])
    x_edges = np.linspace(all_points[:, 0].min(), all_points[:, 0].max(), bins + 1)
    y_edges = np.linspace(all_points[:, 1].min(), all_points[:, 1].max(), bins + 1)

    def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        return float(np.sum(p * np.log(p / q)))

    rows: list[dict[str, object]] = []
    for label in np.unique(labels):
        natural_mask = labels == label
        generated_mask = generated_labels == label
        if not np.any(generated_mask):
            continue
        p_points = data_pca[natural_mask]
        q_points = gen_pca[generated_mask]
        p_hist, _, _ = np.histogram2d(p_points[:, 0], p_points[:, 1], bins=[x_edges, y_edges], density=False)
        q_hist, _, _ = np.histogram2d(q_points[:, 0], q_points[:, 1], bins=[x_edges, y_edges], density=False)
        p = p_hist.ravel().astype(float) + eps
        q = q_hist.ravel().astype(float) + eps
        p /= p.sum()
        q /= q.sum()
        symmetric_kl = 0.5 * (kl_divergence(p, q) + kl_divergence(q, p))
        midpoint = 0.5 * (p + q)
        jensen_shannon = 0.5 * (kl_divergence(p, midpoint) + kl_divergence(q, midpoint))
        rows.append(
            {
                "method_id": method_id,
                "method": ID_TO_DISPLAY[method_id],
                "num_train_samples": num_train_samples,
                "label": label,
                "seed": seed,
                "symmetric_kl": symmetric_kl,
                "jensen_shannon": jensen_shannon,
            }
        )
    return pd.DataFrame(rows)


def collect_kl_results(dataset: str, t1: str, seeds: list[int], train_sizes: list[int], methods: list[str], outputs_root: Path, models_root: Path) -> pd.DataFrame:
    tokens = get_tokens("protein")
    rows: list[pd.DataFrame] = []
    for seed in seeds:
        data_path = natural_dataset_path(outputs_root, dataset, t1, seed, 5000)
        if not data_path.exists():
            continue
        data_one_hot, data_frame = load_one_hot_flat(data_path, tokens)
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(data_one_hot)
        labels = data_frame["true_label"].to_numpy()

        for train_size in train_sizes:
            for method_id in methods:
                path = sample_csv_path(models_root, dataset, t1, seed, method_id, train_size)
                if not path.exists():
                    continue
                gen_one_hot, gen_frame = load_one_hot_flat(path, tokens)
                gen_pca = pca.transform(gen_one_hot)
                rows.append(
                    histogram_divergences_by_label(
                        data_pca,
                        labels,
                        gen_pca,
                        conditioning_labels(gen_frame, method_id, path),
                        method_id,
                        train_size,
                        seed,
                    )
                )
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def plot_divergence_by_label(
    df_results: pd.DataFrame,
    train_sizes: list[int],
    metric: str,
    ylabel: str,
    output_path: Path,
) -> None:
    palette = {
        "true": COLOR_BY_DISPLAY["True"],
        "onlytrain": COLOR_BY_DISPLAY["Train-only"],
        "onehot": COLOR_BY_DISPLAY["One-hot"],
        "rbm": COLOR_BY_DISPLAY["RBM"],
        "esm2": COLOR_BY_DISPLAY["ESM2 650M"],
        "esmc": COLOR_BY_DISPLAY["ESM C 600M"],
        "ankh": COLOR_BY_DISPLAY["Ankh"],
    }
    markers = {
        "true": "o", "onlytrain": "X", "onehot": "s", "rbm": "D",
        "esm2": "^", "esmc": "v", "ankh": "P",
    }
    fig, axes = plt.subplots(1, len(train_sizes), figsize=(4 * len(train_sizes), 4), dpi=220, sharey=True)
    if len(train_sizes) == 1:
        axes = [axes]
    for axis, train_size in zip(axes, train_sizes):
        subset = df_results[df_results["num_train_samples"] == train_size]
        sns.lineplot(
            data=subset,
            x="label",
            y=metric,
            hue="method_id",
            style="method_id",
            dashes=True,
            markers=markers,
            palette=palette,
            errorbar="sd",
            ax=axis,
            legend=axis is axes[0],
        )
        axis.set_xlabel("Label")
        axis.set_ylabel(ylabel)
        axis.set_title(f"{train_size} training samples")
        axis.tick_params(axis="x", rotation=45)
    axes[0].legend(title="")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_average_divergence(df_results: pd.DataFrame, metric: str, ylabel: str, output_path: Path) -> None:
    average_metric = f"average_{metric}"
    summary = (
        df_results.groupby(["num_train_samples", "method", "seed"], as_index=False)[metric]
        .mean()
        .rename(columns={metric: average_metric})
    )
    true_rows = summary[summary["method"] == "True"].copy()
    other_rows = summary[summary["method"] != "True"].copy()

    fig, ax = plt.subplots(figsize=(5, 5), dpi=220)
    sns.lineplot(
        data=other_rows,
        x="num_train_samples",
        y=average_metric,
        hue="method",
        style="method",
        markers=MARKER_BY_DISPLAY,
        dashes=False,
        linewidth=2,
        palette=COLOR_BY_DISPLAY,
        markersize=8,
        ax=ax,
    )
    if not true_rows.empty:
        sns.lineplot(
            data=true_rows,
            x="num_train_samples",
            y=average_metric,
            hue="method",
            style="method",
            dashes=False,
            linewidth=2,
            palette=COLOR_BY_DISPLAY,
            markersize=8,
            legend=True,
            ax=ax,
        )
    ax.set_xlabel("Number of train samples")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    ax.set_xticks([100, 500, 1000, 2000, 5000], labels=[100, 500, 1000, 2000, 5000], rotation=45)
    ax.legend(title="Predictor")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def load_prediction_metrics(path: Path, predictor: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        y_true = decode_array(handle["test"]["labels_true"][()])
        y_pred = decode_array(handle["test"]["predictions"][predictor]["labels_pred"][()])
        y_probs = handle["test"]["predictions"][predictor]["labels_probs"][()]
        class_labels = np.unique(decode_array(handle["train"]["labels_true"][()]))
    return y_true, y_pred, y_probs, class_labels


def collect_self_consistency_metrics(dataset: str, t1: str, seeds: list[int], train_sizes: list[int], methods: list[str], predictor: str, models_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in seeds:
        for train_size in train_sizes:
            for method_id in methods:
                if method_id == "onlytrain":
                    continue
                path = prediction_h5_path(models_root, dataset, t1, seed, method_id, train_size)
                if not path.exists():
                    continue
                y_true, y_pred, y_probs, class_labels = load_prediction_metrics(path, predictor)
                rows.append(
                    {
                        "seed": seed,
                        "num_train_samples": train_size,
                        "method_id": method_id,
                        "method": ID_TO_DISPLAY[method_id],
                        "f1_score": f1_score(y_true, y_pred, average="macro"),
                        "roc_auc": average_multiclass_roc(y_true, y_probs, class_labels)[2],
                    }
                )
    return pd.DataFrame(rows)


def plot_self_consistency_f1(metrics: pd.DataFrame, dataset: str, t1: str, methods: list[str], output_path: Path) -> None:
    order = [ID_TO_DISPLAY[method] for method in methods if method != "onlytrain"]
    fig, ax = plt.subplots(figsize=(6, 5), dpi=220)
    sns.barplot(
        data=metrics,
        x="method",
        y="f1_score",
        hue="num_train_samples",
        order=order,
        palette="YlGnBu",
        edgecolor="black",
        linewidth=1,
        ax=ax,
    )
    ax.set_ylim(0, 1)
    ax.set_title(f"{dataset} - Averaged over {metrics['seed'].nunique()} train/test splits")
    ax.set_xlabel("Method")
    ax.set_ylabel("F1 Score")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_self_consistency_roc(dataset: str, t1: str, seeds: list[int], methods: list[str], predictor: str, train_size: int, models_root: Path, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5), dpi=220)
    for method_id in methods:
        if method_id == "onlytrain":
            continue
        display_name = ID_TO_DISPLAY[method_id]
        all_tpr = []
        all_auc = []
        mean_fpr = np.linspace(0.0, 1.0, 100)
        for seed in seeds:
            path = prediction_h5_path(models_root, dataset, t1, seed, method_id, train_size)
            if not path.exists():
                continue
            y_true, _, y_probs, class_labels = load_prediction_metrics(path, predictor)
            fpr, tpr, auc = average_multiclass_roc(y_true, y_probs, class_labels, npoints=100)
            mean_fpr = fpr
            all_tpr.append(tpr)
            all_auc.append(auc)
        if not all_tpr:
            continue
        all_tpr_array = np.asarray(all_tpr)
        mean_tpr = np.mean(all_tpr_array, axis=0)
        std_tpr = np.std(all_tpr_array, axis=0)
        mean_auc = float(np.mean(all_auc))
        std_auc = float(np.std(all_auc))
        ax.plot(mean_fpr, mean_tpr, color=COLOR_BY_DISPLAY[display_name], label=f"{display_name} (AUC = {mean_auc:.3f} ± {std_auc:.3f})")
        ax.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color=COLOR_BY_DISPLAY[display_name], alpha=0.2)
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random guessing")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    #ax.set_title(f"{dataset} - t1={t1} - {train_size} training samples")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def connected_correlation_statistics(sequences: list[str], tokens: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = encode_sequence(sequences, tokens)
    one_hot = torch.tensor(np.eye(len(tokens))[encoded], dtype=torch.float32)
    return get_freq_single_point(one_hot), get_freq_two_points(one_hot)


def connected_correlation_pearson(
    natural_statistics: tuple[torch.Tensor, torch.Tensor], generated_sequences: list[str], tokens: list[str]
) -> float:
    """Compare all C_ij(a, b) entries for i < j between two aligned MSAs."""
    fi, fij = natural_statistics
    pi, pij = connected_correlation_statistics(generated_sequences, tokens)
    return float(get_correlation_two_points(fij=fij, pij=pij, fi=fi, pi=pi)[0])


def collect_pearson_results(dataset: str, t1: str, seeds: list[int], train_sizes: list[int], methods: list[str], outputs_root: Path, models_root: Path) -> pd.DataFrame:
    tokens = get_tokens("protein")
    rows: list[dict[str, object]] = []
    for seed in seeds:
        data_path = natural_dataset_path(outputs_root, dataset, t1, seed, 5000)
        if not data_path.exists():
            continue
        data_frame = pd.read_csv(data_path)
        natural_sequences = data_frame["sequence_align"].astype(str)
        natural_statistics = connected_correlation_statistics(natural_sequences.tolist(), tokens)
        natural_statistics_by_label = {
            label: connected_correlation_statistics(
                data_frame[data_frame["true_label"].astype(str) == label]["sequence_align"].astype(str).tolist(), tokens
            )
            for label in sorted(set(data_frame["true_label"].astype(str)))
        }

        for train_size in train_sizes:
            for method_id in methods:
                samples_path = sample_csv_path(models_root, dataset, t1, seed, method_id, train_size)
                if not samples_path.exists():
                    continue
                gen_frame = pd.read_csv(samples_path)
                generated_conditioning_labels = conditioning_labels(gen_frame, method_id, samples_path)
                generated_sequences = gen_frame["sequence_align"].astype(str)
                rows.append(
                    {
                        "seed": seed,
                        "num_train_samples": train_size,
                        "method_id": method_id,
                        "method": ID_TO_DISPLAY[method_id],
                        "scope": "global",
                        "label": "all",
                        "n_natural": len(data_frame),
                        "n_generated": len(gen_frame),
                        "pearson": connected_correlation_pearson(
                            natural_statistics, generated_sequences.tolist(), tokens
                        ),
                    }
                )

                natural_labels = set(data_frame["true_label"].astype(str))
                generated_labels = set(generated_conditioning_labels)
                for label in sorted(natural_labels & generated_labels):
                    natural_subset = data_frame[data_frame["true_label"].astype(str) == label]
                    generated_subset = gen_frame[generated_conditioning_labels == label]
                    rows.append(
                        {
                            "seed": seed,
                            "num_train_samples": train_size,
                            "method_id": method_id,
                            "method": ID_TO_DISPLAY[method_id],
                            "scope": "label",
                            "label": label,
                            "n_natural": len(natural_subset),
                            "n_generated": len(generated_subset),
                            "pearson": connected_correlation_pearson(
                                natural_statistics_by_label[label],
                                generated_subset["sequence_align"].astype(str).tolist(),
                                tokens,
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def plot_pearson_correlations(df: pd.DataFrame, train_size: int, output_prefix: Path) -> None:
    if df.empty:
        return
    method_order = [name for name in DISPLAY_TO_ID if name in set(df["method"])]
    df_labels = df[(df["scope"] == "label") & (df["num_train_samples"] == train_size)]
    df_global = df[df["scope"] == "global"]

    if not df_labels.empty:
        fig, ax = plt.subplots(figsize=(max(8, 1.2 * df_labels["label"].nunique()), 5), dpi=220)
        sns.barplot(
            data=df_labels,
            x="label",
            y="pearson",
            hue="method",
            hue_order=method_order,
            palette=COLOR_BY_DISPLAY,
            errorbar="sd",
            edgecolor="black",
            linewidth=0.7,
            ax=ax,
        )
        ax.set_ylim(0, 1)
        ax.set_xlabel("Conditioning label")
        ax.set_ylabel(r"Pearson correlation of $C_{ij}(a,b)$")
        ax.set_title(f"Pairwise connected correlations ({train_size} training samples)")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(title="", loc="upper left", bbox_to_anchor=(1, 1))
        fig.tight_layout()
        fig.savefig(output_prefix.with_name(f"{output_prefix.name}_by_label.pdf"))
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=220)
    sns.lineplot(
        data=df_global,
        x="num_train_samples",
        y="pearson",
        hue="method",
        style="method",
        hue_order=method_order,
        style_order=method_order,
        palette=COLOR_BY_DISPLAY,
        markers=MARKER_BY_DISPLAY,
        dashes=False,
        errorbar="sd",
        ax=ax,
    )
    ax.set_ylim(0, 1)
    ax.set_xscale("log")
    train_sizes = sorted(df_global["num_train_samples"].unique())
    ax.set_xticks(train_sizes, labels=train_sizes)
    ax.set_xlabel("Number of training samples")
    ax.set_ylabel(r"Global Pearson correlation of $C_{ij}(a,b)$")
    ax.legend(title="", loc="best")
    fig.tight_layout()
    fig.savefig(output_prefix.with_name(f"{output_prefix.name}_global.pdf"))
    plt.close(fig)




# Reviewed RR PCA renderer: four predicted-label columns plus a separate True panel.
BUDGETS = (100, 1000)
METHODS = ("onlytrain", "onehot", "rbm", "ankh")
METHOD_TITLES = {
    "true": "True labels",
    "onlytrain": "Train-only labels",
    "onehot": "One-hot labels",
    "rbm": "RBM labels",
    "ankh": "Ankh labels",
}
X_LABEL_FONT_SIZE = 16
Y_LABEL_FONT_SIZE = 16
TICK_FONT_SIZE = 16
LEGEND_FONT_SIZE = 16
PLOT_LEGEND_VERTICAL_GAP = 0.05  # Fraction of figure height; larger moves both legends down.
NATURAL_MARKER_SIZE = 100
GENERATED_MARKER_SIZE = 10
MARGINAL_LINEWIDTH = 2.0
PALETTE = "Dark2"  # Seaborn palette for the specificity colors.
# Orient each independently fitted PCA using the same RR specificity contrasts.
# PC1 points from PF00486 toward PF01339; PC2 toward PF04397.
PCA_ORIENTATION_ANCHORS = (("PF00486", "PF01339"), ("PF00486", "PF04397"))


def load_projection(path: Path, pca: PCA, tokens: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    frame = pd.read_csv(path, dtype={"true_label": str})
    required = {"sequence_align", "true_label"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} must contain {sorted(required)}")
    if frame[list(required)].isna().any().any():
        raise ValueError(f"Missing sequences or true labels in {path}")
    encoded = encode_sequence(frame["sequence_align"].tolist(), tokens)
    if encoded.shape[1] != pca.n_features_in_ // len(tokens):
        raise ValueError(f"Sequence length in {path} differs from the PCA reference")
    one_hot = np.eye(len(tokens), dtype=np.float32)[encoded]
    projected = pca.transform(one_hot.reshape(len(frame), -1))
    return projected, frame["true_label"].to_numpy(dtype=str)


def fit_reference_pca(path: Path, tokens: list[str]) -> tuple[PCA, np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing full natural union: {path}")
    frame = pd.read_csv(path, dtype={"true_label": str})
    required = {"sequence_align", "true_label"}
    if not required.issubset(frame.columns) or frame[list(required)].isna().any().any():
        raise ValueError(f"{path} must have complete sequence_align and true_label columns")
    encoded = encode_sequence(frame["sequence_align"].tolist(), tokens)
    one_hot = np.eye(len(tokens), dtype=np.float32)[encoded].reshape(len(frame), -1)
    # Randomized SVD keeps this visualization practical for the large RR union.
    pca = PCA(n_components=2, svd_solver="randomized", random_state=0)
    projected = pca.fit_transform(one_hot)
    labels = frame["true_label"].to_numpy(dtype=str)
    # The sign of each PCA component is arbitrary. Fix it before projecting any
    # other data so that the same specificities occupy the same sides at each t1.
    for axis, (negative_label, positive_label) in enumerate(PCA_ORIENTATION_ANCHORS):
        negative = projected[labels == negative_label, axis]
        positive = projected[labels == positive_label, axis]
        if not len(negative) or not len(positive):
            raise ValueError(
                f"Cannot orient PC{axis + 1}: {path} needs both "
                f"{negative_label} and {positive_label}"
            )
        if positive.mean() < negative.mean():
            pca.components_[axis] *= -1
            projected[:, axis] *= -1
    return pca, projected, labels


def sample_indices(size: int, maximum: int, rng: np.random.Generator) -> np.ndarray:
    if size <= maximum:
        return np.arange(size)
    return rng.choice(size, size=maximum, replace=False)


def density_curve(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges)
    width = np.diff(edges)
    density = counts / (max(counts.sum(), 1) * width)
    return gaussian_filter1d(density, sigma=0.8)


def validate_label_counts(natural: tuple[np.ndarray, np.ndarray], generated: tuple[np.ndarray, np.ndarray], sample_path: Path) -> None:
    natural_labels, natural_counts = np.unique(natural[1], return_counts=True)
    generated_labels, generated_counts = np.unique(generated[1], return_counts=True)
    if not (np.array_equal(natural_labels, generated_labels) and np.array_equal(natural_counts, generated_counts)):
        raise ValueError(f"Generated label counts in {sample_path} differ from the natural union")


def plot_panel(
    cell,
    natural: tuple[np.ndarray, np.ndarray],
    generated: tuple[np.ndarray, np.ndarray],
    labels: list[str],
    colors: dict[str, tuple[float, float, float]],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    max_points: int,
    rng: np.random.Generator,
    show_x_label: bool,
    show_y_label: bool,
    x_label: str,
    y_label: str,
) -> None:
    inner = cell.subgridspec(2, 2, height_ratios=[1, 4], width_ratios=[4, 1], hspace=0.04, wspace=0.04)
    ax_top = plt.subplot(inner[0, 0])
    ax_joint = plt.subplot(inner[1, 0], sharex=ax_top)
    ax_right = plt.subplot(inner[1, 1], sharey=ax_joint)

    natural_xy, natural_labels = natural
    generated_xy, generated_labels = generated
    # Sample each specificity separately so small classes remain visible.
    for label in labels:
        class_idx = np.flatnonzero(natural_labels == label)
        if not len(class_idx):
            continue
        quota = max(1, round(max_points * len(class_idx) / len(natural_xy)))
        chosen = class_idx[sample_indices(len(class_idx), quota, rng)]
        ax_joint.scatter(
            natural_xy[chosen, 0], natural_xy[chosen, 1],
            s=NATURAL_MARKER_SIZE, c=[colors[label]], alpha=0.23, linewidths=0,
            rasterized=True, zorder=1,
        )
    for label in labels:
        class_idx = np.flatnonzero(generated_labels == label)
        if not len(class_idx):
            continue
        quota = max(1, round(max_points * len(class_idx) / len(generated_xy)))
        chosen = class_idx[sample_indices(len(class_idx), quota, rng)]
        ax_joint.scatter(
            generated_xy[chosen, 0], generated_xy[chosen, 1],
            s=GENERATED_MARKER_SIZE, c=[colors[label]], alpha=1.0,
            edgecolors="black", linewidths=0.18,
            rasterized=True, zorder=2,
        )

    xedges = np.linspace(*xlim, 39)
    yedges = np.linspace(*ylim, 39)
    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])
    for label in labels:
        nat = natural_xy[natural_labels == label]
        gen = generated_xy[generated_labels == label]
        color = colors[label]
        if len(nat):
            ax_top.plot(xcenters, density_curve(nat[:, 0], xedges), color=color, alpha=0.5, lw=MARGINAL_LINEWIDTH, ls="--")
            ax_right.plot(density_curve(nat[:, 1], yedges), ycenters, color=color, alpha=0.5, lw=MARGINAL_LINEWIDTH, ls="--")
        if len(gen):
            ax_top.plot(xcenters, density_curve(gen[:, 0], xedges), color=color, alpha=0.93, lw=MARGINAL_LINEWIDTH)
            ax_right.plot(density_curve(gen[:, 1], yedges), ycenters, color=color, alpha=0.93, lw=MARGINAL_LINEWIDTH)

    ax_joint.set_xlim(*xlim)
    ax_joint.set_ylim(*ylim)
    ax_joint.set_facecolor("#f8fafc")
    #ax_joint.grid(color="#e4e9ef", linewidth=0.6, zorder=0)
    ax_joint.spines[["top", "right"]].set_visible(False)
    ax_joint.tick_params(labelsize=TICK_FONT_SIZE, length=3)
    if show_x_label:
        ax_joint.set_xlabel(x_label, fontsize=X_LABEL_FONT_SIZE)
    else:
        ax_joint.tick_params(labelbottom=False)
    if show_y_label:
        ax_joint.set_ylabel(y_label, fontsize=Y_LABEL_FONT_SIZE)
    else:
        ax_joint.tick_params(labelleft=False)

    for ax in (ax_top, ax_right):
        ax.spines[:].set_visible(False)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax_top.set_xlim(*xlim)
    ax_right.set_ylim(*ylim)


def add_legends(
    fig,
    labels: list[str],
    colors: dict[str, tuple[float, float, float]],
    source_columns: int,
    label_columns: int,
    source_y: float,
    label_y: float,
) -> None:
    source_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[labels[0]],
               markeredgewidth=0, markersize=8, label="Natural sequences"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[labels[0]],
               markeredgecolor="black", markeredgewidth=0.4, markersize=5, label="Generated sequences"),
        Line2D([0], [0], color="#586272", linestyle="--", lw=1.2, label="Natural marginals"),
        Line2D([0], [0], color="#586272", linestyle="-", lw=1.2, label="Generated marginals"),
    ]
    fig.legend(
        handles=source_handles, loc="lower center", bbox_to_anchor=(0.5, source_y),
        ncol=source_columns, frameon=False, fontsize=LEGEND_FONT_SIZE,
    )
    label_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[label],
               markersize=6, label=label)
        for label in labels
    ]
    fig.legend(
        handles=label_handles, loc="lower center", bbox_to_anchor=(0.5, label_y),
        ncol=label_columns, frameon=False, title="Prescribed specificity",
        fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_FONT_SIZE,
    )


def plot_rr_generation_pca(dataset: str, t1: str, seed: int, outputs_root: Path,
                           models_root: Path, output_dir: Path, max_scatter_points: int,
                           use_latex: bool) -> None:
    if dataset != "RR":
        raise ValueError("The four-column PCA comparison is defined for RR only")
    if seed <= 0 or max_scatter_points <= 0:
        raise ValueError("PCA seed and max_scatter_points must be positive")
    output = output_dir / f"RR_t1_{t1}_seed_{seed}_pca_natural_vs_generated_ntrain_100_1000.pdf"
    args = argparse.Namespace(seed=seed, outputs_root=outputs_root, models_root=models_root,
                              output=output, true_output=output.with_name(f"{output.stem}_true_labels.pdf"),
                              max_scatter_points=max_scatter_points, use_latex=use_latex)
    configure_matplotlib(args.use_latex)
    # Keep a shared fit and shared limits for an honest panel-by-panel comparison.
    tokens = list(get_tokens("protein"))
    full_union_path = natural_dataset_path(args.outputs_root, dataset, t1, args.seed, 5000)
    pca, full_natural_xy, full_natural_labels = fit_reference_pca(full_union_path, tokens)
    full_natural = (full_natural_xy, full_natural_labels)

    true_path = sample_csv_path(args.models_root, dataset, t1, args.seed, "true", 5000)
    true_generated = load_projection(true_path, pca, tokens)
    validate_label_counts(full_natural, true_generated, true_path)
    panels: dict[tuple[int, str], tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]] = {}
    all_xy = [full_natural_xy, true_generated[0]]
    for budget in BUDGETS:
        natural_path = natural_dataset_path(args.outputs_root, dataset, t1, args.seed, budget)
        natural = load_projection(natural_path, pca, tokens)
        for method in METHODS:
            sample_path = sample_csv_path(args.models_root, dataset, t1, args.seed, method, budget)
            generated = load_projection(sample_path, pca, tokens)
            if len(generated[0]) != len(natural[0]):
                raise ValueError(
                    f"{sample_path} has {len(generated[0])} rows; expected {len(natural[0])} "
                    f"for the {budget}-annotation natural union"
                )
            validate_label_counts(natural, generated, sample_path)
            panels[(budget, method)] = (natural, generated)
            all_xy.append(generated[0])

    labels = sorted(set(full_natural_labels))
    for (budget, method), (natural, generated) in panels.items():
        unknown = (set(natural[1]) | set(generated[1])) - set(labels)
        if unknown:
            raise ValueError(f"Unknown labels for {method}, budget {budget}: {sorted(unknown)}")
    # The last specificity is drawn last, so assign it the palette's first color.
    palette = sns.color_palette(PALETTE, n_colors=len(labels))[::-1]
    colors = dict(zip(labels, palette))
    stacked = np.concatenate(all_xy, axis=0)
    lower = np.quantile(stacked, 0.002, axis=0)
    upper = np.quantile(stacked, 0.998, axis=0)
    padding = 0.05 * (upper - lower)
    xlim = (float(lower[0] - padding[0]), float(upper[0] + padding[0]))
    ylim = (float(lower[1] - padding[1]), float(upper[1] + padding[1]))

    fig = plt.figure(figsize=(17.2, 10.4), facecolor="white")
    outer = fig.add_gridspec(
        2, 4, left=0.105, right=0.985, bottom=0.33, top=0.90,
        wspace=0.12, hspace=0.18,
    )
    variance = 100 * pca.explained_variance_ratio_
    percent = r"\%" if args.use_latex else "%"
    x_label = f"PC1 ({variance[0]:.1f}{percent} variance)"
    y_label = f"PC2 ({variance[1]:.1f}{percent} variance)"
    rng = np.random.default_rng(7614 + args.seed)
    for row, budget in enumerate(BUDGETS):
        for col, method in enumerate(METHODS):
            plot_panel(
                outer[row, col], *panels[(budget, method)], labels, colors,
                xlim, ylim, args.max_scatter_points, rng,
                show_x_label=row == 1,
                show_y_label=col == 0,
                x_label=x_label, y_label=y_label,
            )
            if row == 0:
                fig.text(
                    outer[row, col].get_position(fig).x0 + 0.5 * outer[row, col].get_position(fig).width,
                    0.94, METHOD_TITLES[method], ha="center", va="center", fontsize=15,
                    fontweight="bold", color="#243247",
                )
        row_pos = outer[row, 0].get_position(fig)
        fig.text(
            0.025, row_pos.y0 + 0.52 * row_pos.height,
            f"{budget:,} annotations", ha="center", va="center", rotation=90,
            fontsize=13, fontweight="bold", color="#243247",
        )

    source_legend_y = 0.24 - PLOT_LEGEND_VERTICAL_GAP
    add_legends(fig, labels, colors, source_columns=4, label_columns=len(labels),
                source_y=source_legend_y, label_y=source_legend_y - 0.10)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=250, facecolor="white")
    plt.close(fig)
    print(f"Saved {args.output}")

    true_fig = plt.figure(figsize=(9.8, 9.0), facecolor="white")
    true_cell = true_fig.add_gridspec(1, 1, left=0.18, right=0.86, bottom=0.36, top=0.86)[0, 0]
    plot_panel(
        true_cell, full_natural, true_generated, labels, colors,
        xlim, ylim, args.max_scatter_points,
        np.random.default_rng(7614 + args.seed),
        show_x_label=True, show_y_label=True,
        x_label=x_label, y_label=y_label,
    )
    true_fig.text(0.5, 0.94, METHOD_TITLES["true"], ha="center", va="center",
                  fontsize=18, fontweight="bold", color="#243247")
    true_source_legend_y = 0.24 - PLOT_LEGEND_VERTICAL_GAP
    add_legends(true_fig, labels, colors, source_columns=2, label_columns=4,
                source_y=true_source_legend_y, label_y=true_source_legend_y - 0.13)
    args.true_output.parent.mkdir(parents=True, exist_ok=True)
    true_fig.savefig(args.true_output, dpi=250, facecolor="white")
    plt.close(true_fig)
    print(f"Saved {args.true_output}")
    print(f"Explained variance: PC1={variance[0]:.2f}%, PC2={variance[1]:.2f}%")


def main() -> None:
    global FALLBACK_MODELS_ROOT
    args = parse_args()
    FALLBACK_MODELS_ROOT = args.models_root
    configure_matplotlib(args.use_latex)
    args.output_dir = args.output_dir or args.outputs_root / "images" / "generation_natural" / args.dataset / f"t1_{args.t1}"
    tables_dir = args.tables_dir or args.outputs_root / "analysis" / "generation_natural" / args.dataset / f"t1_{args.t1}" / "tables"
    artifacts_root = args.samples_root or args.models_root
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    self_images_dir = args.outputs_root / "images" / "generation_self_consistency" / args.dataset / f"t1_{args.t1}"
    self_tables_dir = args.outputs_root / "analysis" / "generation_self_consistency" / args.dataset / f"t1_{args.t1}" / "tables"
    self_images_dir.mkdir(parents=True, exist_ok=True)
    self_tables_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_rr_pca or args.pca_only:
        plot_rr_generation_pca(args.dataset, args.t1, args.pca_seed, args.outputs_root, artifacts_root, args.output_dir, args.max_scatter_points, args.use_latex)
    if args.pca_only:
        return
    # Train-only is a single-seed visual reference, not a quantitative benchmark.
    # Exclude it even if an older runner explicitly passes --methods onlytrain.
    methods = [method for method in args.methods if method != "onlytrain"]
    if not methods:
        raise ValueError("No quantitative methods selected; Train-only is reserved for the separate PCA plot")
    print("Train-only: single-seed PCA reference only; excluded from coverage and quantitative analyses.")

    coverage = artifact_coverage(
        args.dataset, args.t1, args.seeds, args.train_sizes, methods, artifacts_root
    )
    coverage_path = tables_dir / f"{args.dataset}_t1_{args.t1}_generated_artifact_coverage.csv"
    coverage.to_csv(coverage_path, index=False)
    for method_id, group in coverage.groupby("method_id", sort=False):
        predictions = group[group["predictions_expected"]]
        print(
            f"{method_id}: natural samples {int(group['natural_samples_present'].sum())}/{len(group)}; "
            f"predictions {int(predictions['predictions_present'].sum())}/{len(predictions)}"
        )

    kl_results = collect_kl_results(
        args.dataset, args.t1, args.kl_seeds, args.train_sizes, methods, args.outputs_root, artifacts_root
    )
    if not kl_results.empty:
        result_columns = ["method_id", "method", "num_train_samples", "label", "seed"]
        kl_results[result_columns + ["symmetric_kl"]].to_csv(
            tables_dir / f"{args.dataset}_t1_{args.t1}_symmetric_kl_by_label.csv", index=False
        )
        kl_results[result_columns + ["jensen_shannon"]].to_csv(
            tables_dir / f"{args.dataset}_t1_{args.t1}_jensen_shannon_by_label.csv", index=False
        )
        kl_results.to_csv(tables_dir / f"{args.dataset}_t1_{args.t1}_histogram_divergences_by_label.csv", index=False)
        plot_divergence_by_label(
            kl_results,
            args.train_sizes,
            "symmetric_kl",
            "Symmetric KL divergence",
            args.output_dir / f"{args.dataset}_t1_{args.t1}_symmetric_kl_by_label.pdf",
        )
        plot_average_divergence(
            kl_results,
            "symmetric_kl",
            "Average symmetric KL divergence",
            args.output_dir / f"{args.dataset}_t1_{args.t1}_average_symmetric_kl.pdf",
        )
        plot_divergence_by_label(
            kl_results,
            args.train_sizes,
            "jensen_shannon",
            "Jensen–Shannon divergence",
            args.output_dir / f"{args.dataset}_t1_{args.t1}_jensen_shannon_by_label.pdf",
        )
        plot_average_divergence(
            kl_results,
            "jensen_shannon",
            "Average Jensen–Shannon divergence",
            args.output_dir / f"{args.dataset}_t1_{args.t1}_average_jensen_shannon.pdf",
        )

    metrics = collect_self_consistency_metrics(
        args.dataset, args.t1, args.seeds, args.train_sizes, methods, args.predictor, artifacts_root
    )
    if not metrics.empty:
        metrics.to_csv(self_tables_dir / f"{args.dataset}_t1_{args.t1}_generated_sample_metrics.csv", index=False)
        summary = (
            metrics.groupby(["method", "method_id", "num_train_samples"], as_index=False)
            .agg(
                f1_score_mean=("f1_score", "mean"),
                f1_score_std=("f1_score", "std"),
                roc_auc_mean=("roc_auc", "mean"),
                roc_auc_std=("roc_auc", "std"),
            )
        )
        summary.to_csv(self_tables_dir / f"{args.dataset}_t1_{args.t1}_generated_sample_summary.csv", index=False)
        plot_self_consistency_f1(
            metrics,
            args.dataset,
            args.t1,
            methods,
            self_images_dir / f"{args.dataset}_t1_{args.t1}_generated_sample_f1.pdf",
        )
        plot_self_consistency_roc(
            args.dataset,
            args.t1,
            args.seeds,
            methods,
            args.predictor,
            args.roc_train_size,
            artifacts_root,
            self_images_dir / f"{args.dataset}_t1_{args.t1}_generated_sample_roc_n_{args.roc_train_size}.pdf",
        )

    pearson_df = collect_pearson_results(
        args.dataset, args.t1, args.seeds, args.train_sizes, methods, args.outputs_root, artifacts_root
    )
    if not pearson_df.empty:
        pearson_df.to_csv(tables_dir / f"{args.dataset}_t1_{args.t1}_pearson_correlations.csv", index=False)
        plot_pearson_correlations(
            pearson_df,
            args.pearson_train_size,
            args.output_dir / f"{args.dataset}_t1_{args.t1}_pearson_correlations",
        )


if __name__ == "__main__":
    main()
