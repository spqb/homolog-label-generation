#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from adabmDCA.fasta import encode_sequence, get_tokens

from common import REPO_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect train/test split composition and sequence similarity.")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. RR or SH3.")
    parser.add_argument("--t1", required=True, help="Train/test threshold identifier, e.g. 0.4 or 0.7.")
    parser.add_argument("--seed", type=int, required=True, help="Seed identifier.")
    parser.add_argument(
        "--train-sizes",
        type=int,
        nargs="+",
        default=[100, 500, 1000, 2000, 5000],
        help="Train subset sizes to inspect.",
    )
    parser.add_argument(
        "--identity-train-size",
        type=int,
        default=500,
        help="Train subset size used for sequence-identity and PCA diagnostics.",
    )
    parser.add_argument(
        "--identity-label",
        default=None,
        help="Label used for the identity histogram plot. Defaults to the last label encountered, matching the notebook behavior.",
    )
    parser.add_argument(
        "--pca-train-size",
        type=int,
        default=None,
        help="Train subset size used for the PCA projection. Defaults to --identity-train-size.",
    )
    parser.add_argument(
        "--use-latex",
        action="store_true",
        help="Use the same matplotlib LaTeX text configuration as the original notebook.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "images" / "analysis",
        help="Directory where plots are written.",
    )
    return parser.parse_args()


def pairwise_seqid(a: torch.Tensor, b: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
    results = []
    with torch.no_grad():
        for start in range(0, a.shape[0], batch_size):
            batch = a[start : start + batch_size]
            seqid = (batch[:, None, :] == b[None, :, :]).float().mean(dim=-1)
            results.append(seqid.cpu())
    return torch.cat(results, dim=0)


def configure_matplotlib(use_latex: bool) -> None:
    if use_latex:
        plt.rc("text", usetex=True)
        plt.rc("font", size=14)


def plot_label_counts(base_dir: Path, train_sizes: list[int], output_path: Path) -> None:
    label_set: set[str] = set()
    train_counts: dict[int, pd.Series] = {}
    for train_size in train_sizes:
        frame = pd.read_csv(base_dir / "splits" / f"train_{train_size}.csv")
        counts = frame["label"].astype(str).value_counts()
        train_counts[train_size] = counts
        label_set.update(counts.index.tolist())

    labels = sorted(label_set)
    test_counts = pd.read_csv(base_dir / "splits" / "test.csv")["label"].astype(str).value_counts().reindex(labels, fill_value=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=180)
    previous = np.zeros(len(labels), dtype=int)
    for train_size in train_sizes:
        counts = train_counts[train_size].reindex(labels, fill_value=0).to_numpy()
        increment = counts - previous
        axes[0].bar(labels, increment, bottom=previous, label=f"train_{train_size}")
        previous = counts

    axes[0].set_title("Train set (stacked by sample size)")
    axes[0].set_ylabel("Count")
    axes[0].legend(title="Training set size")
    axes[1].bar(labels, test_counts.to_numpy())
    axes[1].set_title("Test label counts")
    axes[1].set_ylabel("Count")
    for axis in axes:
        axis.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_identity_histograms(base_dir: Path, train_size: int, label_to_plot: str | None, output_path: Path) -> None:
    tokens = get_tokens("protein")
    train_frame = pd.read_csv(base_dir / "splits" / f"train_{train_size}.csv")
    test_frame = pd.read_csv(base_dir / "splits" / "test.csv")

    train_encoded = torch.as_tensor(encode_sequence(train_frame["sequence_align"].tolist(), tokens))
    test_encoded = torch.as_tensor(encode_sequence(test_frame["sequence_align"].tolist(), tokens))
    train_labels = train_frame["label"].to_numpy()
    test_labels = test_frame["label"].to_numpy()

    labels = sorted(np.unique(test_labels))
    stats: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for label in labels:
        train_mask = train_labels == label
        test_mask = test_labels == label
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        train_subset = train_encoded[train_mask]
        test_subset = test_encoded[test_mask]

        train_test = pairwise_seqid(test_subset, train_subset).flatten().numpy()
        test_self = pairwise_seqid(test_subset, test_subset)
        test_i, test_j = torch.triu_indices(test_self.shape[0], test_self.shape[1], offset=1)
        test_self_values = test_self[test_i, test_j].numpy()
        train_self = pairwise_seqid(train_subset, train_subset)
        train_i, train_j = torch.triu_indices(train_self.shape[0], train_self.shape[1], offset=1)
        train_self_values = train_self[train_i, train_j].numpy()
        stats[str(label)] = (train_test, test_self_values, train_self_values)

        print(f"\nProcessing label: {label}")
        if train_test.size:
            print(f"Max sequence identity between train and test sets: {train_test.max():.4f}")
        if test_self_values.size:
            print(f"Max sequence identity between test set and itself: {test_self_values.max():.4f}")
        if train_self_values.size:
            print(f"Max sequence identity between train set and itself: {train_self_values.max():.4f}")

    if not stats:
        raise RuntimeError("No label-specific sequence-identity statistics could be computed.")

    selected_label = label_to_plot if label_to_plot is not None else sorted(stats.keys())[-1]
    if selected_label not in stats:
        raise ValueError(f"Requested label '{selected_label}' is not available. Found: {sorted(stats.keys())}")

    train_test, test_self_values, train_self_values = stats[selected_label]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=180)
    axes[0].hist(train_test, bins=50, alpha=0.7, color="blue")
    axes[0].set_title("Train-Test Sequence Identities")
    axes[1].hist(test_self_values, bins=50, alpha=0.7, color="green")
    axes[1].set_title("Test Set Self-Identities")
    axes[2].hist(train_self_values, bins=50, alpha=0.7, color="red")
    axes[2].set_title("Train Set Self-Identities")

    for axis in axes:
        axis.set_xlabel("Sequence Identity")
        axis.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_pca_projection(base_dir: Path, train_size: int, output_path: Path) -> None:
    tokens = get_tokens("protein")
    train_frame = pd.read_csv(base_dir / "splits" / f"train_{train_size}.csv")
    test_frame = pd.read_csv(base_dir / "splits" / "test.csv")

    train_encoded = encode_sequence(train_frame["sequence_align"].tolist(), tokens)
    test_encoded = encode_sequence(test_frame["sequence_align"].tolist(), tokens)

    all_encoded = np.concatenate([train_encoded, test_encoded], axis=0)
    all_binary = np.eye(len(tokens))[all_encoded].reshape(all_encoded.shape[0], -1)
    pca = PCA(n_components=2)
    pca.fit(all_binary)

    train_pca = pca.transform(np.eye(len(tokens))[train_encoded].reshape(train_encoded.shape[0], -1))
    test_pca = pca.transform(np.eye(len(tokens))[test_encoded].reshape(test_encoded.shape[0], -1))

    labels = sorted(np.unique(np.concatenate([train_frame["label"].to_numpy(), test_frame["label"].to_numpy()])))
    if len(labels) > 8:
        raise ValueError("Too many unique labels to plot in a 2x4 grid.")
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), dpi=192, squeeze=False)

    for axis, label in zip(axes.flat, labels):
        train_mask = train_frame["label"].to_numpy() == label
        test_mask = test_frame["label"].to_numpy() == label
        axis.scatter(test_pca[test_mask, 0], test_pca[test_mask, 1], label="Test", alpha=0.5)
        axis.scatter(train_pca[train_mask, 0], train_pca[train_mask, 1], label="Train", alpha=0.9, edgecolors="black", linewidth=0.5)
        axis.set_title(f"Label: {label}")
        axis.legend()

    for axis in axes.flat[len(labels) :]:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_matplotlib(args.use_latex)
    base_dir = REPO_ROOT / "outputs" / args.dataset / f"t1_{args.t1}" / f"seed_{args.seed}"
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    pca_train_size = args.pca_train_size if args.pca_train_size is not None else args.identity_train_size

    plot_label_counts(base_dir, args.train_sizes, output_dir / f"{args.dataset}_t1_{args.t1}_seed_{args.seed}_train_test_counts.pdf")
    plot_identity_histograms(
        base_dir,
        args.identity_train_size,
        args.identity_label,
        output_dir / f"{args.dataset}_t1_{args.t1}_seed_{args.seed}_identity_histograms_ntrain_{args.identity_train_size}.pdf",
    )
    plot_pca_projection(
        base_dir,
        pca_train_size,
        output_dir / f"{args.dataset}_t1_{args.t1}_seed_{args.seed}_pca_projection.pdf",
    )


if __name__ == "__main__":
    main()
