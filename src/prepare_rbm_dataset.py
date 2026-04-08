import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, cast

import h5py
import numpy as np
import pandas as pd


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge prediction HDF5 files into CSV datasets grouped by n_train from info='set-n_train-model'."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory to scan recursively for .h5 files produced by predict_from_embeddings.py.",
    )
    parser.add_argument(
        "--csv_pool",
        type=str,
        required=True,
        help="CSV file containing at least columns 'header' and 'sequence_align'.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where merged CSV files (one per n_train) will be written.",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default="logreg",
        help="Classifier name inside test/predictions used for test labels_pred when multiple are available.",
    )
    return parser


def _decode_scalar(value) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _decode_array(arr: np.ndarray) -> List[str]:
    return [_decode_scalar(x) for x in arr.tolist()]


def _require_dataset(group: h5py.Group, name: str, file_path: Path) -> h5py.Dataset:
    if name not in group:
        raise ValueError(f"Missing '{name}' in {file_path}")
    node = group[name]
    if not isinstance(node, h5py.Dataset):
        raise ValueError(f"'{name}' must be a dataset in {file_path}")
    return node


def _read_info(h5_file: h5py.File, file_path: Path) -> Tuple[str, str, str]:
    if "info" not in h5_file:
        raise ValueError(f"Missing 'info' dataset in {file_path}")

    info_node = h5_file["info"]
    if not isinstance(info_node, h5py.Dataset):
        raise ValueError(f"'info' must be a dataset in {file_path}")

    raw_info = info_node[()]
    info = _decode_scalar(raw_info)
    parts = info.split("-", 2)
    if len(parts) != 3:
        raise ValueError(
            f"Invalid info format in {file_path}: '{info}'. Expected 'set-n_train-model'."
        )

    set_name, n_train, model_name = parts
    return set_name, n_train, model_name


def _choose_classifier(predictions_group: h5py.Group, preferred: str) -> str:
    available = list(predictions_group.keys())
    if not available:
        raise ValueError("No classifier groups found under 'test/predictions'.")

    if preferred in predictions_group:
        return preferred

    for candidate in ["random_forest", "logreg", "SVM"]:
        if candidate in predictions_group:
            return candidate

    return available[0]


def main(args) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pool_df = pd.read_csv(args.csv_pool)
    if "header" not in pool_df.columns or "sequence_align" not in pool_df.columns:
        raise ValueError("csv_pool must contain columns: 'header' and 'sequence_align'.")

    pool_df["header"] = pool_df["header"].astype(str)
    pool_df["sequence_align"] = pool_df["sequence_align"].astype(str)
    header_to_sequence: Dict[str, str] = dict(zip(pool_df["header"], pool_df["sequence_align"]))

    h5_paths = sorted(input_dir.rglob("*.h5"))
    if not h5_paths:
        raise ValueError(f"No .h5 files found under {input_dir}")

    grouped_rows: Dict[str, Dict[Tuple[str, str], Dict[str, str]]] = defaultdict(dict)
    grouped_models: Dict[str, set] = defaultdict(set)

    for h5_path in h5_paths:
        with h5py.File(h5_path, "r") as f:
            _, n_train, model_name = _read_info(f, h5_path)
            grouped_models[n_train].add(model_name)
            model_col = f"{model_name}_label"

            if "train" not in f or "test" not in f:
                raise ValueError(f"Missing 'train' or 'test' group in {h5_path}")

            train_group = f["train"]
            test_group = f["test"]
            if not isinstance(train_group, h5py.Group) or not isinstance(test_group, h5py.Group):
                raise ValueError(f"Invalid 'train'/'test' group structure in {h5_path}")

            train_headers = _decode_array(cast(np.ndarray, _require_dataset(train_group, "headers", h5_path)[:]))
            train_labels = _decode_array(cast(np.ndarray, _require_dataset(train_group, "labels_true", h5_path)[:]))

            test_headers = _decode_array(cast(np.ndarray, _require_dataset(test_group, "headers", h5_path)[:]))
            test_labels = (
                _decode_array(cast(np.ndarray, _require_dataset(test_group, "labels_true", h5_path)[:]))
                if "labels_true" in test_group
                else [""] * len(test_headers)
            )

            if "predictions" not in test_group:
                raise ValueError(f"Missing 'test/predictions' in {h5_path}")
            predictions_group = test_group["predictions"]
            if not isinstance(predictions_group, h5py.Group):
                raise ValueError(f"'test/predictions' is not a group in {h5_path}")

            classifier_name = _choose_classifier(predictions_group, args.classifier)
            classifier_group = predictions_group[classifier_name]
            if not isinstance(classifier_group, h5py.Group):
                raise ValueError(f"Classifier group '{classifier_name}' is invalid in {h5_path}")
            pred_labels = _decode_array(cast(np.ndarray, _require_dataset(classifier_group, "labels_pred", h5_path)[:]))

            if len(test_headers) != len(pred_labels):
                raise ValueError(f"Mismatch between test headers and predictions in {h5_path}")

            # Fill train rows: model column must copy true_label.
            for header, true_label in zip(train_headers, train_labels):
                key = ("train", header)
                row = grouped_rows[n_train].setdefault(
                    key,
                    {
                        "header": header,
                        "set": "train",
                        "true_label": true_label,
                        "train_only": true_label,
                    },
                )

                if row["true_label"] != true_label:
                    raise ValueError(
                        f"Inconsistent true_label for train header '{header}' in n_train={n_train}"
                    )

                row[model_col] = true_label

            # Fill test rows: model column gets predicted label.
            for header, true_label, pred_label in zip(test_headers, test_labels, pred_labels):
                key = ("test", header)
                row = grouped_rows[n_train].setdefault(
                    key,
                    {
                        "header": header,
                        "set": "test",
                        "true_label": true_label,
                        "train_only": "",
                    },
                )

                if row["true_label"] != true_label:
                    raise ValueError(
                        f"Inconsistent true_label for test header '{header}' in n_train={n_train}"
                    )

                row[model_col] = pred_label

    for n_train, rows_dict in grouped_rows.items():
        model_cols = [f"{m}_label" for m in sorted(grouped_models[n_train])]
        rows: List[Dict[str, str]] = []

        for (_, header), row in sorted(rows_dict.items(), key=lambda x: (x[0][0], x[0][1])):
            out_row: Dict[str, str] = {
                "header": header,
                "sequence_align": header_to_sequence.get(header, ""),
                "set": row["set"],
                "true_label": row["true_label"],
            }
            for col in model_cols:
                out_row[col] = row.get(col, "")
            out_row["train_only"] = row["true_label"] if row["set"] == "train" else ""
            rows.append(out_row)

        out_df = pd.DataFrame(rows)
        ordered_cols = ["header", "sequence_align", "set", "true_label", *model_cols, "train_only"]
        out_df = out_df[ordered_cols]

        output_file = output_dir / f"rbm_dataset_ntrain_{n_train}.csv"
        out_df.to_csv(output_file, index=False)
        print(f"Wrote {output_file} ({len(out_df)} rows)")


if __name__ == "__main__":
    parser = get_parser()
    main(parser.parse_args())
