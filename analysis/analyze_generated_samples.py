#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

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
}
ID_TO_DISPLAY = {value: key for key, value in DISPLAY_TO_ID.items()}
COLOR_BY_DISPLAY = {
    "True": "#0072b2",
    "Train-only": "#7a7a7a",
    "One-hot": "#f96363",
    "RBM": "#ffb400",
    "ESM2 650M": "#00b170",
    "ESM C 600M": "#8e63ce",
}
MARKER_BY_DISPLAY = {
    "True": "o",
    "Train-only": "X",
    "One-hot": "s",
    "RBM": "D",
    "ESM2 650M": "^",
    "ESM C 600M": "v",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the main generated-sample analyses from sample_rbm.ipynb.")
    parser.add_argument("--dataset", default="RR")
    parser.add_argument("--t1", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    parser.add_argument("--kl-seeds", type=int, nargs="+", default=[1, 3, 4, 5, 6, 7, 8, 9, 10])
    parser.add_argument("--train-sizes", type=int, nargs="+", default=[100, 500, 1000, 2000, 5000])
    parser.add_argument("--predictor", default="logreg")
    parser.add_argument("--roc-train-size", type=int, default=100)
    parser.add_argument("--pca-seed", type=int, default=1)
    parser.add_argument("--pca-train-size", type=int, default=100)
    parser.add_argument("--pearson-train-size", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "images" / "analysis")
    parser.add_argument("--use-latex", action="store_true")
    return parser.parse_args()


def configure_matplotlib(use_latex: bool) -> None:
    if use_latex:
        plt.rcParams.update({"text.usetex": True, "font.size": 14})


def natural_dataset_path(dataset: str, t1: str, seed: int, train_size: int) -> Path:
    return REPO_ROOT / "outputs" / dataset / f"t1_{t1}" / f"seed_{seed}" / "datasets_for_rbm" / f"rbm_dataset_ntrain_{train_size}.csv"


def sample_dir(dataset: str, t1: str, seed: int, method_id: str, train_size: int) -> Path:
    if method_id == "true":
        return REPO_ROOT / "models" / dataset / f"t1_{t1}" / f"seed_{seed}" / "RBM_labels_all_true"
    return REPO_ROOT / "models" / dataset / f"t1_{t1}" / f"seed_{seed}" / f"RBM_labels_{train_size}_{method_id}"


def sample_csv_path(dataset: str, t1: str, seed: int, method_id: str, train_size: int) -> Path:
    return sample_dir(dataset, t1, seed, method_id, train_size) / "samples.csv"


def prediction_h5_path(dataset: str, t1: str, seed: int, method_id: str, train_size: int) -> Path:
    suffix = "true" if method_id == "true" else method_id
    return sample_dir(dataset, t1, seed, method_id, train_size) / f"samples.predictions.{suffix}.h5"


def load_one_hot_flat(csv_path: Path, tokens: list[str], sequence_column: str = "sequence_align") -> tuple[np.ndarray, pd.DataFrame]:
    frame = pd.read_csv(csv_path)
    encoded = encode_sequence(frame[sequence_column].tolist(), tokens)
    one_hot = np.eye(len(tokens))[encoded]
    return one_hot.reshape(len(one_hot), -1), frame


def average_multiclass_roc(y_true: np.ndarray, y_probs: np.ndarray, npoints: int = 100) -> tuple[np.ndarray, np.ndarray, float]:
    labels = np.unique(y_true)
    probs = np.asarray(y_probs)
    if probs.ndim == 1:
        probs = probs[:, None]
    if len(labels) == 2 and probs.shape[1] == 1:
        probs = np.column_stack([1.0 - probs[:, 0], probs[:, 0]])

    grid = np.linspace(0.0, 1.0, npoints)
    mean_tpr = np.zeros_like(grid)
    auc_values = []
    for index, label in enumerate(labels):
        y_bin = (y_true == label).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, probs[:, index])
        mean_tpr += np.interp(grid, fpr, tpr)
        auc_values.append(roc_auc_score(y_bin, probs[:, index]))
    mean_tpr /= len(labels)
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


def plot_scatter(ax, data: np.ndarray, generated: np.ndarray, labels: np.ndarray, colors: list) -> None:
    for index, label in enumerate(np.unique(labels)):
        mask = labels == label
        ax.scatter(data[mask, 0], data[mask, 1], color=colors[index], alpha=0.4, s=80, label=str(label))
        ax.scatter(generated[mask, 0], generated[mask, 1], color=colors[index], s=30, edgecolor="black", linewidth=0.3)


def plot_hist(ax, data: np.ndarray, generated: np.ndarray, labels: np.ndarray, colors: list, orientation: str = "vertical") -> None:
    idx = 0 if orientation == "vertical" else 1
    for index, label in enumerate(np.unique(labels)):
        mask = labels == label
        ax.hist(data[mask, idx], bins=30, orientation=orientation, color=colors[index], alpha=1.0, density=True, histtype="step")
        ax.hist(generated[mask, idx], bins=30, orientation=orientation, color=colors[index], alpha=0.4, density=True)


def plot_conditional_generation_panel(dataset: str, t1: str, seed: int, train_size: int, output_path: Path) -> None:
    tokens = get_tokens("protein")
    data_path = natural_dataset_path(dataset, t1, seed, 5000)
    if not data_path.exists():
        return

    data_one_hot, data_frame = load_one_hot_flat(data_path, tokens)
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_one_hot)
    pc1_variance, pc2_variance = 100.0 * pca.explained_variance_ratio_
    y_true = data_frame["true_label"].to_numpy()

    methods = ["true", "onlytrain", "onehot", "rbm", "esm2", "esmc"]
    generated_pca: dict[str, np.ndarray] = {}
    for method_id in methods:
        path = sample_csv_path(dataset, t1, seed, method_id, train_size)
        if not path.exists():
            continue
        gen_one_hot, _ = load_one_hot_flat(path, tokens)
        generated_pca[method_id] = pca.transform(gen_one_hot)

    if not generated_pca:
        return

    colors = sns.color_palette("Spectral", len(np.unique(y_true)))
    display_titles = {
        "true": "True Labels",
        "onlytrain": "Train-only Labels",
        "onehot": "Onehot Labels",
        "rbm": "RBM Labels",
        "esm2": "ESM2 650M Labels",
        "esmc": "ESM C 600M Labels",
    }

    n_preds = len(generated_pca)
    width_ratios = []
    for _ in range(n_preds):
        width_ratios.extend([0.2, 1.0])
    width_ratios.append(0.2)

    fig = plt.figure(figsize=(5.5 * n_preds, 5.5), dpi=300)
    grid = fig.add_gridspec(2, 2 * n_preds + 1, width_ratios=width_ratios, height_ratios=[0.2, 1], wspace=0.1, hspace=0.1)

    for index, method_id in enumerate(generated_pca):
        scatter_ax = fig.add_subplot(grid[1, 2 * index + 1])
        histx_ax = fig.add_subplot(grid[0, 2 * index + 1], sharex=scatter_ax)
        histy_ax = fig.add_subplot(grid[1, 2 * index + 2], sharey=scatter_ax)

        plot_scatter(scatter_ax, data_pca, generated_pca[method_id], y_true, colors)
        plot_hist(histx_ax, data_pca, generated_pca[method_id], y_true, colors)
        plot_hist(histy_ax, data_pca, generated_pca[method_id], y_true, colors, orientation="horizontal")

        scatter_ax.set_xlabel(f"PC1 ({pc1_variance:.1f}% variance)")
        if index == 0:
            scatter_ax.set_ylabel(f"PC2 ({pc2_variance:.1f}% variance)")
        else:
            scatter_ax.set_yticklabels([])
        histx_ax.set_title(display_titles.get(method_id, method_id))
        histx_ax.axis("off")
        histy_ax.axis("off")

    fig.suptitle(f"Variance explained by PC1 + PC2: {pc1_variance + pc2_variance:.1f}%", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def collect_kl_results(dataset: str, t1: str, seeds: list[int], train_sizes: list[int]) -> pd.DataFrame:
    tokens = get_tokens("protein")
    rows: list[pd.DataFrame] = []
    for seed in seeds:
        data_path = natural_dataset_path(dataset, t1, seed, 5000)
        if not data_path.exists():
            continue
        data_one_hot, data_frame = load_one_hot_flat(data_path, tokens)
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(data_one_hot)
        labels = data_frame["true_label"].to_numpy()

        for train_size in train_sizes:
            for method_id in ["true", "onehot", "rbm", "esm2", "esmc"]:
                path = sample_csv_path(dataset, t1, seed, method_id, train_size)
                if not path.exists():
                    continue
                gen_one_hot, gen_frame = load_one_hot_flat(path, tokens)
                if "true_label" not in gen_frame:
                    raise ValueError(f"Missing conditioning-label column 'true_label' in {path}")
                gen_pca = pca.transform(gen_one_hot)
                rows.append(
                    histogram_divergences_by_label(
                        data_pca,
                        labels,
                        gen_pca,
                        gen_frame["true_label"].to_numpy(),
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
        "onehot": COLOR_BY_DISPLAY["One-hot"],
        "rbm": COLOR_BY_DISPLAY["RBM"],
        "esm2": COLOR_BY_DISPLAY["ESM2 650M"],
        "esmc": COLOR_BY_DISPLAY["ESM C 600M"],
    }
    markers = {"true": "o", "onehot": "s", "rbm": "D", "esm2": "^", "esmc": "v"}
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


def load_prediction_metrics(path: Path, predictor: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        y_true = decode_array(handle["test"]["labels_true"][()])
        y_pred = decode_array(handle["test"]["predictions"][predictor]["labels_pred"][()])
        y_probs = handle["test"]["predictions"][predictor]["labels_probs"][()]
    return y_true, y_pred, y_probs


def collect_self_consistency_metrics(dataset: str, t1: str, seeds: list[int], train_sizes: list[int], predictor: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in seeds:
        for train_size in train_sizes:
            for method_id in ["true", "onehot", "rbm", "esm2", "esmc"]:
                path = prediction_h5_path(dataset, t1, seed, method_id, train_size)
                if not path.exists():
                    continue
                y_true, y_pred, y_probs = load_prediction_metrics(path, predictor)
                rows.append(
                    {
                        "seed": seed,
                        "num_train_samples": train_size,
                        "method_id": method_id,
                        "method": ID_TO_DISPLAY[method_id],
                        "f1_score": f1_score(y_true, y_pred, average="macro"),
                        "roc_auc": average_multiclass_roc(y_true, y_probs)[2],
                    }
                )
    return pd.DataFrame(rows)


def plot_self_consistency_f1(metrics: pd.DataFrame, dataset: str, t1: str, output_path: Path) -> None:
    order = ["True", "One-hot", "RBM", "ESM2 650M", "ESM C 600M"]
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


def plot_self_consistency_roc(dataset: str, t1: str, seeds: list[int], predictor: str, train_size: int, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5), dpi=220)
    for display_name in ["True", "One-hot", "RBM", "ESM2 650M", "ESM C 600M"]:
        method_id = DISPLAY_TO_ID[display_name]
        all_tpr = []
        all_auc = []
        mean_fpr = np.linspace(0.0, 1.0, 100)
        for seed in seeds:
            path = prediction_h5_path(dataset, t1, seed, method_id, train_size)
            if not path.exists():
                continue
            y_true, _, y_probs = load_prediction_metrics(path, predictor)
            fpr, tpr, auc = average_multiclass_roc(y_true, y_probs, npoints=100)
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
    ax.set_title(f"{dataset} - t1={t1} - {train_size} training samples")
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


def collect_pearson_results(dataset: str, t1: str, seeds: list[int], train_sizes: list[int]) -> pd.DataFrame:
    tokens = get_tokens("protein")
    rows: list[dict[str, object]] = []
    for seed in seeds:
        data_path = natural_dataset_path(dataset, t1, seed, 5000)
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
            for method_id in ["true", "onlytrain", "onehot", "rbm", "esm2", "esmc"]:
                samples_path = sample_csv_path(dataset, t1, seed, method_id, train_size)
                if not samples_path.exists():
                    continue
                gen_frame = pd.read_csv(samples_path)
                if "true_label" not in gen_frame:
                    raise ValueError(f"Missing conditioning-label column 'true_label' in {samples_path}")
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
                generated_labels = set(gen_frame["true_label"].astype(str))
                for label in sorted(natural_labels & generated_labels):
                    natural_subset = data_frame[data_frame["true_label"].astype(str) == label]
                    generated_subset = gen_frame[gen_frame["true_label"].astype(str) == label]
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
        ax.set_ylim(-1, 1)
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
    ax.set_ylim(-1, 1)
    ax.set_xscale("log")
    train_sizes = sorted(df_global["num_train_samples"].unique())
    ax.set_xticks(train_sizes, labels=train_sizes)
    ax.set_xlabel("Number of training samples")
    ax.set_ylabel(r"Global Pearson correlation of $C_{ij}(a,b)$")
    ax.legend(title="", loc="best")
    fig.tight_layout()
    fig.savefig(output_prefix.with_name(f"{output_prefix.name}_global.pdf"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_matplotlib(args.use_latex)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_conditional_generation_panel(
        args.dataset,
        args.t1,
        args.pca_seed,
        args.pca_train_size,
        args.output_dir / f"{args.dataset}_t1_{args.t1}_conditional_generations_seed_{args.pca_seed}_n_{args.pca_train_size}.png",
    )

    kl_results = collect_kl_results(args.dataset, args.t1, args.kl_seeds, args.train_sizes)
    if not kl_results.empty:
        result_columns = ["method_id", "method", "num_train_samples", "label", "seed"]
        kl_results[result_columns + ["symmetric_kl"]].to_csv(
            args.output_dir / f"{args.dataset}_t1_{args.t1}_symmetric_kl_by_label.csv", index=False
        )
        kl_results[result_columns + ["jensen_shannon"]].to_csv(
            args.output_dir / f"{args.dataset}_t1_{args.t1}_jensen_shannon_by_label.csv", index=False
        )
        kl_results.to_csv(args.output_dir / f"{args.dataset}_t1_{args.t1}_histogram_divergences_by_label.csv", index=False)
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

    metrics = collect_self_consistency_metrics(args.dataset, args.t1, args.seeds, args.train_sizes, args.predictor)
    if not metrics.empty:
        metrics.to_csv(args.output_dir / f"{args.dataset}_t1_{args.t1}_generated_sample_metrics.csv", index=False)
        summary = (
            metrics.groupby(["method", "method_id", "num_train_samples"], as_index=False)
            .agg(
                f1_score_mean=("f1_score", "mean"),
                f1_score_std=("f1_score", "std"),
                roc_auc_mean=("roc_auc", "mean"),
                roc_auc_std=("roc_auc", "std"),
            )
        )
        summary.to_csv(args.output_dir / f"{args.dataset}_t1_{args.t1}_generated_sample_summary.csv", index=False)
        plot_self_consistency_f1(metrics, args.dataset, args.t1, args.output_dir / f"{args.dataset}_t1_{args.t1}_generated_sample_f1.pdf")
        plot_self_consistency_roc(
            args.dataset,
            args.t1,
            args.seeds,
            args.predictor,
            args.roc_train_size,
            args.output_dir / f"{args.dataset}_t1_{args.t1}_generated_sample_roc_n_{args.roc_train_size}.pdf",
        )

    pearson_df = collect_pearson_results(args.dataset, args.t1, args.seeds, args.train_sizes)
    if not pearson_df.empty:
        pearson_df.to_csv(args.output_dir / f"{args.dataset}_t1_{args.t1}_pearson_correlations.csv", index=False)
        plot_pearson_correlations(
            pearson_df,
            args.pearson_train_size,
            args.output_dir / f"{args.dataset}_t1_{args.t1}_pearson_correlations",
        )


if __name__ == "__main__":
    main()
