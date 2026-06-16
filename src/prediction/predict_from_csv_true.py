#!/usr/bin/env python3

import argparse
import os
import sys
import time
from typing import Optional, Tuple, cast

import h5py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from adabmDCA.fasta import get_tokens, encode_sequence
from utils import load_query_data


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a logistic regression on a CSV dataset and predict on "
            "samples.embedding.true.h5 files under query_source/seed_*/RBM_labels_all_true."
        )
    )
    parser.add_argument("--train_csv", type=str, required=True, help="Path to the training dataset in .csv format.")
    parser.add_argument(
        "--query_source",
        type=str,
        default=None,
        help="Base folder containing seed_<i>/RBM_labels_all_true.",
    )
    parser.add_argument(
        "--query_dir",
        type=str,
        default=None,
        help="Single RBM_labels_all_true folder to process.",
    )
    parser.add_argument("--column_sequences", type=str, default="sequence_align", help="CSV column containing sequences.")
    parser.add_argument("--column_labels", type=str, default="label", help="CSV column containing labels.")
    parser.add_argument("--column_headers", type=str, default="header", help="CSV column containing headers.")
    parser.add_argument("--info", type=str, default="", help="Optional metadata string saved in output .h5 files.")
    return parser


def _normalize_labels(y: np.ndarray) -> np.ndarray:
    if y.dtype.kind == "S":
        return y.astype("U")
    if y.dtype.kind == "O":
        flat = y.reshape(-1)
        if all(isinstance(v, (bytes, bytearray)) for v in flat):
            return np.array([bytes(v).decode("utf-8", errors="replace") for v in flat], dtype="U").reshape(y.shape)
    return y


def _onehot_encode_sequences(sequences: list[str], tokens: list[str]) -> np.ndarray:
    encoded = encode_sequence(sequences, tokens)
    n_sequences, sequence_length = encoded.shape
    n_tokens = len(tokens)
    onehot = np.eye(n_tokens)[encoded]
    return onehot.reshape(n_sequences, sequence_length * n_tokens)


def _load_embeddings_h5(path: str) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    with h5py.File(path, "r") as f:
        X_node = f["embeddings"]
        h_node = f["headers"]
        if not isinstance(X_node, h5py.Dataset) or not isinstance(h_node, h5py.Dataset):
            raise ValueError(f"Invalid embeddings file: {path}. 'embeddings' and 'headers' must be datasets.")

        X = cast(np.ndarray, X_node[:])
        h = cast(np.ndarray, h_node[:])

        y: Optional[np.ndarray] = None
        if "labels" in f:
            y_node = f["labels"]
            if not isinstance(y_node, h5py.Dataset):
                raise ValueError(f"Invalid embeddings file: {path}. 'labels' must be a dataset when present.")
            y = cast(np.ndarray, y_node[:])
            y = _normalize_labels(y)
    return X, y, h


def _save_predictions(
    output_path: str,
    info: str,
    train_headers: np.ndarray,
    train_labels: np.ndarray,
    test_headers: np.ndarray,
    test_labels: Optional[np.ndarray],
    labels_pred: np.ndarray,
    labels_probs: np.ndarray,
) -> None:
    if output_path.split(".")[-1].lower() != "h5":
        output_path += ".h5"

    with h5py.File(output_path, "w") as f:
        f.create_dataset("info", data=np.asarray(info, dtype="S"))
        train_group = f.create_group("train")
        train_group.create_dataset("headers", data=train_headers.astype("S"))
        train_group.create_dataset(
            "labels_true",
            data=train_labels.astype("S") if train_labels.dtype.kind in ("U", "O") else train_labels,
        )

        test_group = f.create_group("test")
        test_group.create_dataset("headers", data=test_headers.astype("S"))
        if test_labels is not None:
            test_group.create_dataset(
                "labels_true",
                data=test_labels.astype("S") if test_labels.dtype.kind in ("U", "O") else test_labels,
            )

        pred_group = test_group.create_group("predictions/logreg")
        if labels_pred.dtype.kind in ("U", "O"):
            labels_pred = labels_pred.astype("S")
        pred_group.create_dataset("labels_pred", data=labels_pred)
        pred_group.create_dataset("labels_probs", data=labels_probs)


def main() -> None:
    args = get_parser().parse_args()

    if not os.path.exists(args.train_csv):
        raise FileNotFoundError(f"Training CSV not found: {args.train_csv}")
    if not args.query_source and not args.query_dir:
        raise ValueError("Either --query_source or --query_dir is required.")
    if args.query_source and args.query_dir:
        raise ValueError("Use only one of --query_source or --query_dir.")
    if args.query_source and not os.path.isdir(args.query_source):
        raise NotADirectoryError(f"query_source does not exist: {args.query_source}")
    if args.query_dir and not os.path.isdir(args.query_dir):
        raise NotADirectoryError(f"query_dir does not exist: {args.query_dir}")

    tokens = get_tokens("protein")

    sequences, headers, labels = load_query_data(
        csv_file=args.train_csv,
        column_sequences=args.column_sequences,
        column_headers=args.column_headers,
        column_labels=args.column_labels,
    )

    if labels is None:
        raise ValueError("Training CSV must contain labels.")

    X_train = _onehot_encode_sequences(sequences, tokens)
    y_train = _normalize_labels(np.asarray(labels))
    train_headers = np.asarray(headers)

    print(f"Loaded {len(sequences)} training sequences.")
    print(f"Training embeddings shape: {X_train.shape}")

    print("Standardizing training data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    print("Training logistic regression...")
    start_time = time.time()
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train_scaled, y_train)
    print(f"Logistic regression trained in {time.time() - start_time:.2f}s")

    if args.query_dir:
        label_dirs = [args.query_dir]
    else:
        seed_dirs = [
            os.path.join(args.query_source, d)
            for d in os.listdir(args.query_source)
            if d.startswith("seed_") and os.path.isdir(os.path.join(args.query_source, d))
        ]

        if not seed_dirs:
            raise FileNotFoundError(f"No seed_* folders found in: {args.query_source}")

        label_dirs = []
        for seed_dir in sorted(seed_dirs):
            label_dir = os.path.join(seed_dir, "RBM_labels_all_true")
            if not os.path.isdir(label_dir):
                print(f"Skipping {os.path.basename(seed_dir)}: RBM_labels_all_true not found.")
                continue
            label_dirs.append(label_dir)

    num_runs = 0
    for label_dir in label_dirs:
        test_h5 = os.path.join(label_dir, "samples.embedding.true.h5")
        output_path = os.path.join(label_dir, "samples.predictions.true.h5")

        if not os.path.isfile(test_h5):
            print(f"Skipping {label_dir}: missing {test_h5}")
            continue

        X_test, y_test, h_test = _load_embeddings_h5(test_h5)
        X_test_scaled = scaler.transform(X_test)

        labels_pred = logreg.predict(X_test_scaled)
        labels_probs = logreg.predict_proba(X_test_scaled)

        info = args.info or f"train_csv={args.train_csv}"
        _save_predictions(
            output_path=output_path,
            info=info,
            train_headers=train_headers,
            train_labels=y_train,
            test_headers=h_test,
            test_labels=y_test,
            labels_pred=labels_pred,
            labels_probs=labels_probs,
        )

        print(f"Saved predictions to {output_path}")
        num_runs += 1

    if num_runs == 0:
        raise RuntimeError("No prediction runs were executed.")

    print(f"Completed {num_runs} prediction runs.")


if __name__ == "__main__":
    main()
