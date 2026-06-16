from __future__ import annotations

from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, roc_curve


REPO_ROOT = Path(__file__).resolve().parents[1]


METHOD_LABELS = {
    "onehot": "One-hot",
    "rbm": "RBM",
    "plm": "Foundation",
    "plm_supervised": "Fine-tuned",
    "true": "True",
    "onlytrain": "Train-only",
}


def decode_array(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind == "S":
        return values.astype("U")
    if values.dtype.kind == "O":
        flat = values.reshape(-1)
        if all(isinstance(v, (bytes, bytearray)) for v in flat):
            return np.array([bytes(v).decode("utf-8", errors="replace") for v in flat], dtype="U").reshape(values.shape)
    return values


def load_prediction_file(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        y_true = decode_array(handle["test"]["labels_true"][()])
        predictor_group = handle["test"]["predictions"]
        predictor_names = list(predictor_group.keys())
        if not predictor_names:
            raise ValueError(f"No predictor groups found in {path}")

        loaded: dict[str, np.ndarray] = {"labels_true": y_true}
        for predictor_name in predictor_names:
            group = predictor_group[predictor_name]
            loaded[f"{predictor_name}/labels_pred"] = decode_array(group["labels_pred"][()])
            loaded[f"{predictor_name}/labels_probs"] = group["labels_probs"][()]
        return loaded


def macro_roc_auc(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    labels = np.unique(y_true)
    if len(labels) < 2:
        return float("nan")

    y_bin = np.column_stack([(y_true == label).astype(int) for label in labels])
    probs = np.asarray(y_probs)
    if probs.ndim == 1:
        probs = probs[:, None]

    if len(labels) == 2 and probs.shape[1] == 1:
        probs = np.column_stack([1.0 - probs[:, 0], probs[:, 0]])

    if probs.shape[1] != len(labels):
        raise ValueError(f"Expected {len(labels)} probability columns, found {probs.shape[1]}")

    auc_values = []
    for index in range(len(labels)):
        auc_values.append(roc_auc_score(y_bin[:, index], probs[:, index]))
    return float(np.mean(auc_values))


def average_multiclass_roc(y_true: np.ndarray, y_probs: np.ndarray, n_points: int = 200) -> tuple[np.ndarray, np.ndarray, float]:
    labels = np.unique(y_true)
    probs = np.asarray(y_probs)
    if probs.ndim == 1:
        probs = probs[:, None]

    if len(labels) == 2 and probs.shape[1] == 1:
        probs = np.column_stack([1.0 - probs[:, 0], probs[:, 0]])

    grid = np.linspace(0.0, 1.0, n_points)
    tpr_sum = np.zeros_like(grid)
    auc_values = []

    for index, label in enumerate(labels):
        y_bin = (y_true == label).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, probs[:, index])
        tpr_sum += np.interp(grid, fpr, tpr)
        auc_values.append(roc_auc_score(y_bin, probs[:, index]))

    mean_tpr = tpr_sum / len(labels)
    mean_auc = float(np.mean(auc_values))
    return grid, mean_tpr, mean_auc


def find_prediction_files(predictions_dir: Path) -> list[Path]:
    return sorted(predictions_dir.glob("*.predictions.h5"))


def parse_test_prediction_name(path: Path) -> tuple[str, int] | None:
    name = path.name
    prefix = "test.embedding."
    suffix = ".predictions.h5"
    if not (name.startswith(prefix) and name.endswith(suffix)):
        return None

    body = name[len(prefix) : -len(suffix)]
    marker = ".ntrain_"
    if marker not in body:
        return None
    model, n_train = body.rsplit(marker, 1)
    try:
        return model, int(n_train)
    except ValueError:
        return None


def predictions_to_dataframe(paths: Iterable[Path], predictor: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in paths:
        parsed = parse_test_prediction_name(path)
        if parsed is None:
            continue
        model_id, n_train = parsed
        loaded = load_prediction_file(path)
        pred_key = f"{predictor}/labels_pred"
        probs_key = f"{predictor}/labels_probs"
        if pred_key not in loaded or probs_key not in loaded:
            continue
        rows.append(
            {
                "path": str(path),
                "model_id": model_id,
                "method": METHOD_LABELS.get(model_id, model_id),
                "num_train_samples": n_train,
                "f1_score": f1_score(loaded["labels_true"], loaded[pred_key], average="macro"),
                "roc_auc": macro_roc_auc(loaded["labels_true"], loaded[probs_key]),
            }
        )
    return pd.DataFrame(rows)
