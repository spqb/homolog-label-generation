#!/usr/bin/env python3
"""Move legacy figures and missing-label results out of source directories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
IMAGE_SUFFIXES = {".pdf", ".png", ".svg", ".jpg", ".jpeg", ".tif", ".tiff"}


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def move(source: Path, target: Path, apply: bool) -> None:
    if target.exists():
        if source.is_file() and target.is_file() and digest(source) == digest(target):
            print(f"ALREADY COPIED {source} -> {target}")
            return
        raise FileExistsError(f"Destination already exists: {target}")
    print(f"MOVE {source.relative_to(ROOT)} -> {target.relative_to(ROOT)}")
    if apply:
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)


def migrate_missing_labels(apply: bool) -> None:
    old = ROOT / "analysis" / "missing_labels"
    if not old.exists():
        return
    for budget in (100, 1000):
        run = old / f"results-n{budget}"
        if not run.exists():
            continue
        for manifest in sorted(run.glob("seed_*/manifest.json")):
            content = json.loads(manifest.read_text())
            data = Path(content["data"])
            if not data.is_file() or digest(data) != content["data_sha256"]:
                raise ValueError(f"Cannot validate missing-label reference: {manifest}")
        for path in sorted(run.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(run)
            if relative.parts[0] == "analysis":
                if path.suffix.lower() in IMAGE_SUFFIXES:
                    target = ROOT / "outputs/images/missing_labels/RR/t1_0.4" / f"ntrain_{budget}_{path.name}"
                else:
                    target = ROOT / "outputs/analysis/missing_labels/RR/t1_0.4" / f"ntrain_{budget}/tables" / path.name
            elif len(relative.parts) >= 3 and relative.parts[1] in {"zero", "posterior"} and path.name != "samples.csv":
                target = ROOT / "models/missing_labels/RR/t1_0.4" / f"ntrain_{budget}" / relative
            else:
                target = ROOT / "outputs/missing_labels/RR/t1_0.4" / f"ntrain_{budget}" / relative
            move(path, target, apply)
    summary = old / "results-summary"
    if summary.exists():
        for path in sorted(summary.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() in IMAGE_SUFFIXES:
                target = ROOT / "outputs/images/missing_labels/RR/t1_0.4" / path.name
            else:
                target = ROOT / "outputs/analysis/missing_labels/RR/t1_0.4/combined/tables" / path.name
            move(path, target, apply)
    if apply:
        for folder in sorted(old.rglob("*"), reverse=True):
            if folder.is_dir() and not any(folder.iterdir()):
                folder.rmdir()


def migrate_images(apply: bool) -> None:
    old = ROOT / "images"
    if not old.is_dir():
        return
    for path in sorted(old.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(old)
        if path.suffix.lower() in IMAGE_SUFFIXES:
            target = ROOT / "outputs/images/legacy" / relative
        else:
            target = ROOT / "outputs/analysis/legacy" / relative
        move(path, target, apply)
    if apply:
        for folder in sorted(old.rglob("*"), reverse=True):
            if folder.is_dir() and not any(folder.iterdir()):
                folder.rmdir()
        if not any(old.iterdir()):
            old.rmdir()


def migrate_generated(apply: bool) -> None:
    for params in sorted((ROOT / "models").glob("*/t1_*/seed_*/*/params.h5")):
        model = params.parent
        valid = model.name == "RBM_labels_all_true_full" or bool(re.fullmatch(r"RBM_labels_\d+_(true|onlytrain|onehot|rbm|ankh|esm2|esmc)", model.name))
        legacy = model.name.startswith("incorrect_RBM_labels_") or model.name.startswith("_RBM_labels_") or model.name == "RBM_labels_all_true"
        if not valid and not legacy:
            continue
        case = model.parent
        kind = "generation" if valid else "generation_legacy"
        destination = ROOT / "outputs" / case.parent.parent.name / case.parent.name / case.name / kind / model.name
        for path in sorted(model.iterdir()):
            if path.is_file() and (path.name.startswith("samples") or path.name.startswith("targets") or path.name.startswith(".samples.")):
                move(path, destination / path.name, apply)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Perform validated moves; default is a dry plan.")
    args = parser.parse_args()
    migrate_missing_labels(args.apply)
    migrate_images(args.apply)
    migrate_generated(args.apply)


if __name__ == "__main__":
    main()
