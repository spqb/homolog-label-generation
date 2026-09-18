#!/usr/bin/env python3
"""Run selected experiment stages with one configuration and artifact inventory."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import tomllib

from pipeline_state import file_identity, inspect, record, signature


ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "src" / "pipeline"
MANIFEST = PIPELINE / "experiments.toml"
STAGES = {
    1: "01_split_dataset.sh", 2: "02_embed_splits.sh",
    3: "03_predict_test_embeddings.sh", 4: "04_train_rr_supervised_plm.sh",
    5: "05_prepare_rbm_datasets.sh", 6: "06_train_conditioned_rbms.sh",
    7: "07_sample_conditioned_rbms.sh", 8: "08_predict_generated_samples.sh",
}
STAGE_NAMES = {"split": 1, "embed": 2, "predict": 3, "supervised": 4,
               "rbm-dataset": 5, "train": 6, "sample": 7, "generated-predict": 8,
               "analyze": 9}
TRAIN_SIZES = (100, 500, 1000, 2000, 5000)
PLAN_ROWS: list[tuple[str, int, str]] = []


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, value: str) -> int:
        self.terminal.write(value)
        self.log.write(value)
        return len(value)

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()


def report(state: str, stage: int, label: str) -> None:
    PLAN_ROWS.append((state, stage, label))
    print(f"{state:12} stage={stage} {label}", flush=True)


def parse_stages(values: list[str] | None, default: list[int]) -> list[int]:
    if not values:
        return default
    selected: set[int] = set()
    for value in values:
        for item in value.split(","):
            if "-" in item and item.replace("-", "").isdigit():
                lo, hi = map(int, item.split("-", 1))
                selected.update(range(lo, hi + 1))
            else:
                selected.add(STAGE_NAMES.get(item, int(item) if item.isdigit() else -1))
    if not selected or not selected <= set(range(1, 10)):
        raise ValueError(f"Invalid stages: {values}")
    return sorted(selected)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=["main", "classification", "annotation-refresh", "generation",
                                         "generation-pca", "missing-labels", "smoke"], default="main")
    p.add_argument("--datasets", nargs="+")
    p.add_argument("--t1", nargs="+")
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--train-sizes", nargs="+", type=int)
    p.add_argument("--methods", nargs="+", help="Classification representations.")
    p.add_argument("--generation-methods", nargs="+", help="Conditioned generators; True is always included.")
    p.add_argument("--stages", nargs="+")
    p.add_argument("--force", nargs="+", default=[])
    p.add_argument("--outputs-root", "--outputs_root", type=Path, default=ROOT / "outputs")
    p.add_argument("--models-root", "--models_root", type=Path, default=ROOT / "models")
    p.add_argument("--plan", "--dry-run", "--dry_run", action="store_true")
    p.add_argument("--use-latex", "--use_latex", action="store_true")
    p.add_argument("--annadca-bin", default="annadca")
    p.add_argument("--jobs", type=int, default=1, help="Reserved for future parallel execution; currently 1.")
    return p


def command(args: list[str], plan: bool) -> None:
    print("Running:", shlex.join(args), flush=True)
    if not plan:
        process = subprocess.Popen(args, cwd=ROOT, start_new_session=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
            returncode = process.wait()
            if returncode:
                raise subprocess.CalledProcessError(returncode, args)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def stage_script(number: int) -> Path:
    return PIPELINE / "stages" / STAGES[number]


def archive(path: Path) -> None:
    if path.exists():
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        archive_path = path.with_name(f"{path.name}.replaced_{stamp}")
        os.replace(path, archive_path)
        print(f"Archived previous artifact: {archive_path}")


def install_files(temp_dir: Path, destination: Path, names: list[str]) -> None:
    for name in names:
        new = temp_dir / name
        marker = name.startswith(".samples.") and name.endswith(".complete")
        if not new.is_file() or (new.stat().st_size == 0 and not marker):
            raise RuntimeError(f"Stage did not create a nonempty artifact: {new}")
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        target = destination / name
        archive(target)
        os.replace(temp_dir / name, target)


def run_file_stage(number: int, label: str, inputs: list[Path], outputs: list[Path],
                   settings: dict, arguments: list[str], output_flag: str,
                   plan: bool, force: bool, allow_legacy: bool) -> str:
    script = stage_script(number)
    expected = signature([script, *inputs], settings)
    legacy_fresh = allow_legacy and all(path.is_file() for path in [*inputs, *outputs]) and (
        min(path.stat().st_mtime_ns for path in outputs) >=
        max(path.stat().st_mtime_ns for path in inputs)
    )
    state = inspect(outputs, outputs[0], expected, legacy_fresh)
    if force and state != "BLOCKED":
        state = "STALE" if any(path.exists() for path in outputs) else "BUILD"
    report(state, number, label)
    if state == "BLOCKED":
        if not plan:
            missing = [str(path) for path in inputs if not path.is_file()]
            raise FileNotFoundError(f"Missing stage {number} inputs: {missing}")
        return state
    if state in {"REUSE", "LEGACY_REUSE"}:
        if state == "LEGACY_REUSE" and not plan:
            record(outputs[0], expected, outputs, legacy=True, inputs=[script, *inputs], settings=settings)
        return state
    if plan:
        return state
    destination = outputs[0].parent
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".stage_{number}_", dir=destination.parent) as temporary:
        temp_dir = Path(temporary)
        command(["bash", str(script), *arguments, output_flag, str(temp_dir)], False)
        install_files(temp_dir, destination, [path.name for path in outputs])
    record(outputs[0], expected, outputs, inputs=[script, *inputs], settings=settings)
    return state


def method_model_path(case_model: Path, method: str, budget: int) -> Path:
    name = "RBM_labels_all_true_full" if method == "true" else f"RBM_labels_{budget}_{method}"
    return case_model / name


def generation_methods(dataset: str, seed: int, requested: list[str]) -> list[str]:
    if dataset != "RR":
        return []
    selected = list(dict.fromkeys(["true", *requested]))
    if dataset == "RR" and seed == 1:
        selected.append("onlytrain")
    return list(dict.fromkeys(selected))


def write_case_manifest(case: Path, source: Path, embedding_rbm: Path, dataset: str,
                        t1: str, seed: int, budgets: list[int], methods: list[str],
                        generators: list[str], gibbs_steps: int | None,
                        samples_per_label: int) -> None:
    """Record the selected experiment and actual labeled subset sizes."""
    actual_sizes: dict[str, int] = {}
    for budget in budgets:
        split = case / "splits" / f"train_{budget}.csv"
        if split.is_file():
            with split.open(newline="") as stream:
                actual_sizes[str(budget)] = sum(1 for _ in csv.DictReader(stream))
    payload = {"dataset": dataset, "t1": t1, "seed": seed,
               "source": file_identity(source), "embedding_rbm": file_identity(embedding_rbm),
               "nominal_train_sizes": budgets, "actual_train_sizes": actual_sizes,
               "classification_methods": methods, "generation_methods": generators,
               "generation_gibbs_steps": gibbs_steps,
               "self_consistency_per_label": samples_per_label}
    case.mkdir(parents=True, exist_ok=True)
    target = case / "manifest.json"
    temporary = target.with_name(target.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, target)


def run_case(dataset: str, t1: str, seed: int, config: dict, defaults: dict,
             stages: list[int], methods: list[str], generators: list[str],
             budgets: list[int], args: argparse.Namespace) -> bool:
    source = ROOT / config["source_csv"]
    embedding_rbm = args.models_root / dataset / Path(config["embedding_rbm"]).name
    case = args.outputs_root / dataset / f"t1_{t1}" / f"seed_{seed}"
    splits, embeddings, predictions = (case / name for name in ("splits", "embeddings", "predictions"))
    rbm_datasets = case / "datasets_for_rbm"
    model_dir = args.models_root / dataset / f"t1_{t1}" / f"seed_{seed}"
    sample_dir = case / "generation"
    allow_legacy = dataset != "cm"
    force = set(args.force)
    case_generators = generation_methods(dataset, seed, generators)
    frozen_methods = [method for method in methods if method != "esm2_supervised"]
    split_files = [splits / "test.csv", *(splits / f"train_{n}.csv" for n in budgets)]

    def embedding_spec(method: str) -> tuple[list[Path], list[Path], dict]:
        output_files = [embeddings / f"test.embedding.{method}.h5",
                        *(embeddings / f"train_{n}.embedding.{method}.h5" for n in budgets)]
        encoder = ROOT / ("src/encoding/plm_encoding.py" if method in {"esm2", "esmc", "ankh"}
                          else f"src/encoding/{method}_encoding.py")
        inputs = [*split_files, encoder, *([embedding_rbm] if method == "rbm" else [])]
        settings = {"method": method, "model_id": defaults.get(f"{method}_model", ""),
                    "max_length": defaults["max_length"]}
        return output_files, inputs, settings

    def prediction_spec(method: str) -> tuple[list[Path], list[Path], dict]:
        embedding_files = embedding_spec(method)[0]
        output_files = [predictions / f"test.embedding.{method}.ntrain_{n}.predictions.h5" for n in budgets]
        inputs = [*embedding_files, ROOT / "src/predict_from_embeddings.py"]
        return output_files, inputs, {"method": method, "budgets": budgets}

    print(f"\nCase {dataset} t1={t1} seed={seed}", flush=True)
    if not source.is_file() or not embedding_rbm.is_file():
        raise FileNotFoundError(f"Missing source or embedding RBM: {source}, {embedding_rbm}")

    if 1 in stages:
        run_file_stage(1, f"{dataset}/{t1}/{seed} splits", [source, ROOT / "src/split_train_test.py"], split_files,
                       {"seed": seed, "t1": t1, "budgets": budgets, "source": str(source)},
                       ["--source_csv", str(source), "--seed", str(seed), "--t1", t1], "--output_dir",
                       args.plan, "split" in force or "1" in force, allow_legacy)
    elif dataset == "cm" and any(stage in stages for stage in (2, 3, 5, 6, 7, 8, 9)):
        split_state = inspect(split_files, split_files[0], signature([stage_script(1), source, ROOT / "src/split_train_test.py"],
                            {"seed": seed, "t1": t1, "budgets": budgets, "source": str(source)}), False)
        if split_state != "REUSE":
            report("BLOCKED", 1, f"cm/{t1}/{seed} merged-source Cobalt splits")
            if args.plan:
                return False
            raise RuntimeError(
                "cm splits have no matching completion record for data/cm/cm_sequences_merged.csv. "
                "Stage 9 alone cannot verify the splits or rebuild their downstream predictions. "
                f"Run src/pipeline/run_experiments.sh --profile classification --datasets cm --t1 {t1} "
                "without --stages 9 first."
            )

    if 2 in stages:
        for method in frozen_methods:
            output_files, inputs, settings = embedding_spec(method)
            run_file_stage(2, f"{dataset}/{t1}/{seed} {method} embeddings", inputs, output_files,
                           settings,
                           ["--splits_dir", str(splits), "--models", method,
                            "--rbm_model_path", str(embedding_rbm), "--esm2_model", defaults["esm2_model"],
                            "--esmc_model", defaults["esmc_model"], "--ankh_model", defaults["ankh_model"],
                            "--max_length", str(defaults["max_length"])], "--output_dir",
                           args.plan, "embed" in force or "2" in force, allow_legacy)

    if 3 in stages:
        for method in frozen_methods:
            outputs, inputs, settings = prediction_spec(method)
            run_file_stage(3, f"{dataset}/{t1}/{seed} {method} predictions",
                           inputs, outputs, settings,
                           ["--embeddings_dir", str(embeddings), "--models", method], "--predictions_dir",
                           args.plan, "predict" in force or "3" in force, allow_legacy)

    if dataset == "cm" and 9 in stages and not {1, 2, 3} <= set(stages):
        for method in frozen_methods:
            for number, (outputs, inputs, settings) in (
                (2, embedding_spec(method)), (3, prediction_spec(method))
            ):
                expected = signature([stage_script(number), *inputs], settings)
                if inspect(outputs, outputs[0], expected) != "REUSE":
                    report("BLOCKED", number, f"cm/{t1}/{seed} {method} classification inputs")
                    if args.plan:
                        return False
                    raise RuntimeError(
                        f"cm seed {seed} has unverified {method} stage-{number} artifacts. "
                        f"Run src/pipeline/run_experiments.sh --profile classification --datasets cm "
                        f"--t1 {t1} --seeds {seed} without --stages 9 before analyzing this seed."
                    )

    if 4 in stages and config.get("supervised_plm") and (args.methods is None or "esm2_supervised" in args.methods):
        supervised = [predictions / f"test.embedding.esm2_supervised.ntrain_{n}.predictions.h5"
                      for n in budgets if n >= 1000]
        expected = signature([stage_script(4), *split_files], {"min_ntrain": 1000, "dataset": dataset})
        state = inspect(supervised, supervised[0], expected, allow_legacy)
        if "supervised" in force or "4" in force:
            state = "STALE"
        report(state, 4, f"{dataset}/{t1}/{seed} supervised ESM2")
        if state == "BLOCKED" and not args.plan:
            raise FileNotFoundError("Missing RR supervised PLM split inputs")
        if state in {"BUILD", "STALE"} and not args.plan:
            command(["bash", str(stage_script(4)), "--dataset", dataset, "--splits_dir", str(splits),
                     "--predictions_dir", str(predictions), "--model_dir", str(model_dir),
                     "--seed", str(seed), "--enabled", "1", "--min_ntrain", "1000"], False)
            record(supervised[0], expected, supervised,
                   inputs=[stage_script(4), *split_files],
                   settings={"min_ntrain": 1000, "dataset": dataset})

    if 5 in stages:
        predictor_methods = ["onehot", "rbm", "ankh"]
        pred_files = [predictions / f"test.embedding.{method}.ntrain_{n}.predictions.h5"
                      for method in predictor_methods for n in budgets]
        outputs = [rbm_datasets / f"rbm_dataset_ntrain_{n}.csv" for n in budgets]
        run_file_stage(5, f"{dataset}/{t1}/{seed} RBM unions", [source, *pred_files, ROOT / "src/prepare_rbm_dataset.py"],
                       outputs, {"methods": predictor_methods, "budgets": budgets},
                       ["--predictions_dir", str(predictions), "--source_csv", str(source),
                        "--models", *predictor_methods, "--train_sizes", *map(str, budgets)], "--output_dir",
                       args.plan, "rbm-dataset" in force or "5" in force, allow_legacy)

    if 6 in stages:
        for method in case_generators:
            model_budgets = [max(budgets)] if method == "true" else budgets
            outputs = [method_model_path(model_dir, method, n) / "params.h5" for n in model_budgets]
            inputs = [rbm_datasets / f"rbm_dataset_ntrain_{n}.csv" for n in model_budgets]
            expected = signature([stage_script(6), *inputs], {"method": method, "updates": 50000, "seed": seed})
            legacy_checkpoint_ok = allow_legacy
            if legacy_checkpoint_ok:
                try:
                    import h5py
                    for checkpoint in outputs:
                        with h5py.File(checkpoint, "r") as handle:
                            required_update = "update_50000" if method != "onlytrain" else None
                            if (required_update and required_update not in handle) or (method == "onlytrain" and not any(key.startswith("update_") for key in handle)):
                                legacy_checkpoint_ok = False
                except (OSError, FileNotFoundError):
                    legacy_checkpoint_ok = False
            state = inspect(outputs, outputs[0], expected, legacy_checkpoint_ok)
            if "train" in force or "6" in force:
                state = "STALE"
            report(state, 6, f"{dataset}/{t1}/{seed} {method} RBM")
            if state in {"REUSE", "LEGACY_REUSE"}:
                if state == "LEGACY_REUSE" and not args.plan:
                    record(outputs[0], expected, outputs, legacy=True,
                           inputs=[stage_script(6), *inputs],
                           settings={"method": method, "updates": 50000, "seed": seed})
                continue
            if state == "BLOCKED":
                if not args.plan:
                    raise FileNotFoundError(f"Missing RBM union for {method}")
                continue
            if args.plan:
                continue
            model_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".rbm_train_", dir=model_dir) as temporary:
                temp = Path(temporary)
                command(["bash", str(stage_script(6)), "--datasets_dir", str(rbm_datasets),
                         "--model_dir", str(temp), "--annadca_bin", args.annadca_bin,
                         "--methods", method, "--train_sizes", *map(str, budgets)], False)
                for output in outputs:
                    replacement = temp / output.parent.name
                    if not (replacement / "params.h5").is_file():
                        raise RuntimeError(f"No completed checkpoint: {replacement}")
                for output in outputs:
                    archive(output.parent)
                    os.replace(temp / output.parent.name, output.parent)
            record(outputs[0], expected, outputs,
                   inputs=[stage_script(6), *inputs],
                   settings={"method": method, "updates": 50000, "seed": seed})

    if 7 in stages:
        for method in case_generators:
            model_budgets = [max(budgets)] if method == "true" else budgets
            names = [method_model_path(model_dir, method, n).name for n in model_budgets]
            input_models = [model_dir / name / "params.h5" for name in names]
            input_unions = [rbm_datasets / f"rbm_dataset_ntrain_{n}.csv" for n in model_budgets]
            gibbs = int(config["gibbs_steps"])
            completed_names = ["samples.true_labels.csv", f".samples.true_labels.gibbs_{gibbs}.complete"]
            if method != "onlytrain":
                completed_names += ["targets.self_consistency.csv", "samples.self_consistency.csv",
                                    f".samples.self_consistency.gibbs_{gibbs}.complete"]
            outputs = [sample_dir / name / filename for name in names for filename in completed_names]
            expected = signature([stage_script(7), *input_models, *input_unions],
                                 {"method": method, "gibbs": gibbs, "per_label": defaults["samples_per_label"]})
            legacy_samples_ok = allow_legacy and all(
                (sample_dir / name / f".samples.true_labels.gibbs_{gibbs}.complete").is_file()
                and (method == "onlytrain" or (sample_dir / name / f".samples.self_consistency.gibbs_{gibbs}.complete").is_file())
                for name in names
            ) and all(path.is_file() for path in [*outputs, *input_models, *input_unions]) and (
                min(path.stat().st_mtime_ns for path in outputs) >=
                max(path.stat().st_mtime_ns for path in [*input_models, *input_unions])
            )
            state = inspect(outputs, outputs[0], expected, legacy_samples_ok)
            if "sample" in force or "7" in force:
                state = "STALE"
            report(state, 7, f"{dataset}/{t1}/{seed} {method} samples")
            if state in {"REUSE", "LEGACY_REUSE"}:
                if state == "LEGACY_REUSE" and not args.plan:
                    record(outputs[0], expected, outputs, legacy=True,
                           inputs=[stage_script(7), *input_models, *input_unions],
                           settings={"method": method, "gibbs": gibbs,
                                     "per_label": defaults["samples_per_label"]})
                continue
            if state == "BLOCKED":
                if not args.plan:
                    raise FileNotFoundError(f"Missing conditioned RBM or union for {method}")
                continue
            if args.plan:
                continue
            sample_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".samples_", dir=sample_dir) as temporary:
                temp = Path(temporary)
                command(["bash", str(stage_script(7)), "--datasets_dir", str(rbm_datasets),
                         "--model_dir", str(model_dir), "--samples_dir", str(temp),
                         "--annadca_bin", args.annadca_bin, "--methods", method,
                         "--train_sizes", *map(str, budgets), "--gibbs_steps", str(gibbs),
                         "--self_consistency_per_label", str(defaults["samples_per_label"])], False)
                for name in names:
                    install_files(temp / name, sample_dir / name, completed_names)
            record(outputs[0], expected, outputs,
                   inputs=[stage_script(7), *input_models, *input_unions],
                   settings={"method": method, "gibbs": gibbs,
                             "per_label": defaults["samples_per_label"]})

    if 8 in stages:
        for method in (m for m in case_generators if m != "onlytrain"):
            model_budgets = [max(budgets)] if method == "true" else budgets
            names = [method_model_path(model_dir, method, n).name for n in model_budgets]
            samples = [sample_dir / name / "samples.self_consistency.csv" for name in names]
            outputs = [sample_dir / name / f"samples.self_consistency.predictions.{method}.h5" for name in names]
            expected = signature([stage_script(8), *samples, source, embedding_rbm],
                                 {"method": method, "max_length": defaults["max_length"]})
            legacy_predictions_ok = allow_legacy and all(
                output.is_file() and sample.is_file() and output.stat().st_mtime_ns >= sample.stat().st_mtime_ns
                for output, sample in zip(outputs, samples)
            )
            state = inspect(outputs, outputs[0], expected, legacy_predictions_ok)
            if "generated-predict" in force or "8" in force:
                state = "STALE"
            report(state, 8, f"{dataset}/{t1}/{seed} {method} predictions")
            if state in {"REUSE", "LEGACY_REUSE"}:
                if state == "LEGACY_REUSE" and not args.plan:
                    record(outputs[0], expected, outputs, legacy=True,
                           inputs=[stage_script(8), *samples, source, embedding_rbm],
                           settings={"method": method, "max_length": defaults["max_length"]})
                continue
            if state == "BLOCKED":
                if not args.plan:
                    raise FileNotFoundError(f"Missing self-consistency samples for {method}")
                continue
            if args.plan:
                continue
            command(["bash", str(stage_script(8)), "--source_csv", str(source),
                     "--splits_dir", str(splits), "--embeddings_dir", str(embeddings),
                     "--model_dir", str(model_dir), "--samples_dir", str(sample_dir),
                     "--rbm_model_path", str(embedding_rbm), "--methods", method,
                     "--train_sizes", *map(str, budgets), "--max_length", str(defaults["max_length"]),
                     "--ankh_model", defaults["ankh_model"], "--esm2_model", defaults["esm2_model"],
                     "--esmc_model", defaults["esmc_model"]], False)
            record(outputs[0], expected, outputs,
                   inputs=[stage_script(8), *samples, source, embedding_rbm],
                   settings={"method": method, "max_length": defaults["max_length"]})

    if not args.plan:
        write_case_manifest(case, source, embedding_rbm, dataset, t1, seed, budgets,
                            methods, case_generators,
                            int(config["gibbs_steps"]) if config["generation_enabled"] else None,
                            int(defaults["samples_per_label"]))

    if 9 in stages and args.profile in {"main", "smoke"}:
        report("BUILD", 9, f"{dataset}/{t1}/{seed} split diagnostics")
        command(["bash", str(PIPELINE / "stages/09_analyze.sh"), "splits",
                 "--dataset", dataset, "--t1", t1, "--seed", str(seed),
                 "--outputs-root", str(args.outputs_root),
                 "--train-sizes", *map(str, budgets),
                 *(["--use-latex"] if args.use_latex else [])], args.plan)
    return True


def run_analysis(dataset: str, t1: str, seeds: list[int], methods: list[str],
                 generators: list[str], budgets: list[int], stages: list[int],
                 args: argparse.Namespace) -> None:
    if 9 not in stages:
        return
    latex = ["--use-latex"] if args.use_latex else []
    if args.profile in {"main", "classification", "annotation-refresh", "smoke"}:
        report("BUILD", 9, f"{dataset}/{t1} classification analysis")
        base_methods = ["onehot", "rbm", "esm2", "esmc", "ankh"] if args.profile == "annotation-refresh" else methods
        include_supervised = dataset == "RR" and (args.profile == "annotation-refresh" or args.methods is None or "esm2_supervised" in methods)
        analysis_methods = [*base_methods, *(["esm2_supervised"] if include_supervised else [])]
        command(["bash", str(PIPELINE / "stages/09_analyze.sh"), "classification",
                 "--dataset", dataset, "--t1", t1, "--seeds", *map(str, seeds),
                 "--train-sizes", *map(str, budgets), "--methods", *analysis_methods,
                 "--outputs-root", str(args.outputs_root), "--plot-roc-labels", *latex], args.plan)
    if dataset == "RR" and args.profile in {"main", "generation", "generation-pca"}:
        report("BUILD", 9, f"{dataset}/{t1} generation analysis")
        pca_only = args.profile == "generation-pca"
        invocation = ["bash", str(PIPELINE / "stages/09_analyze.sh"), "generation-pca" if pca_only else "generation",
                      "--dataset", dataset, "--t1", t1, "--seeds", *map(str, seeds),
                      "--kl-seeds", *map(str, seeds), "--train-sizes", *map(str, budgets),
                      "--methods", "true", *[m for m in generators if m != "true"],
                      "--outputs-root", str(args.outputs_root), "--models-root", str(args.models_root),
                      "--samples-root", str(args.outputs_root), *latex]
        if dataset == "RR" and 1 in seeds:
            invocation += ["--plot-rr-pca", "--pca-seed", "1"]
        if pca_only:
            invocation.append("--pca-only")
        command(invocation, args.plan)


def run_missing_labels(args: argparse.Namespace, defaults: dict, profile: dict) -> None:
    datasets = args.datasets or profile["datasets"]
    if datasets != ["RR"]:
        raise ValueError("The missing-label comparison is configured for RR only")
    t1_values = args.t1 or profile.get("t1", ["0.4"])
    seeds = args.seeds or profile.get("seeds", [1, 2, 3])
    budgets = args.train_sizes or profile.get("train_sizes", [100, 1000])
    for t1 in t1_values:
        for budget in budgets:
            complete = True
            for seed in seeds:
                case = args.outputs_root / "missing_labels" / "RR" / f"t1_{t1}" / f"ntrain_{budget}" / f"seed_{seed}"
                manifest = case / "manifest.json"
                if not manifest.is_file():
                    complete = False
                    break
                meta = json.loads(manifest.read_text())
                reference = Path(meta["data"])
                if not reference.is_file() or file_identity(reference)["sha256"] != meta["data_sha256"]:
                    complete = False
                    break
                for arm in ("zero", "posterior"):
                    params = args.models_root / "missing_labels" / "RR" / f"t1_{t1}" / f"ntrain_{budget}" / f"seed_{seed}" / arm / "params.h5"
                    samples = case / arm / "samples.csv"
                    if not params.is_file() or not samples.is_file() or params.stat().st_size == 0 or samples.stat().st_size == 0:
                        complete = False
            report("REUSE" if complete else "BUILD", 6, f"RR/{t1} missing-label arms ntrain={budget}")
            command([sys.executable, str(PIPELINE / "missing_labels.py"),
                     "--dataset", "RR", "--t1", t1, "--train-size", str(budget),
                     "--seeds", *map(str, seeds), "--outputs-root", str(args.outputs_root),
                     "--models-root", str(args.models_root / "missing_labels"), "--skip-analysis"], args.plan)
            report("BUILD", 9, f"RR/{t1} missing-label analysis ntrain={budget}")
            command(["bash", str(PIPELINE / "stages/09_analyze.sh"), "missing-labels",
                     "--run-dir", str(args.outputs_root / "missing_labels" / "RR" / f"t1_{t1}" / f"ntrain_{budget}"),
                     "--seeds", *map(str, seeds),
                     "--tables-dir", str(args.outputs_root / "analysis" / "missing_labels" / "RR" / f"t1_{t1}" / f"ntrain_{budget}" / "tables"),
                     "--images-dir", str(args.outputs_root / "images" / "missing_labels" / "RR" / f"t1_{t1}")], args.plan)
        if set(budgets) >= {100, 1000}:
            report("BUILD", 9, f"RR/{t1} missing-label combined figure")
            command(["bash", str(PIPELINE / "stages/09_analyze.sh"), "missing-labels-summary",
                     "--runs", *[str(args.outputs_root / "missing_labels" / "RR" / f"t1_{t1}" / f"ntrain_{budget}") for budget in budgets],
                     "--output-dir", str(args.outputs_root / "analysis" / "missing_labels" / "RR" / f"t1_{t1}" / "combined" / "tables"),
                     "--images-dir", str(args.outputs_root / "images" / "missing_labels" / "RR" / f"t1_{t1}")], args.plan)


def execute(args: argparse.Namespace, p: argparse.ArgumentParser, config: dict) -> None:
    defaults = config["defaults"]
    profile = config["profiles"][args.profile]
    if args.profile == "missing-labels":
        run_missing_labels(args, defaults, profile)
        return
    datasets = args.datasets or profile["datasets"]
    budgets = args.train_sizes or profile.get("train_sizes", defaults["train_sizes"])
    methods = args.methods or defaults["classification_methods"]
    generators = args.generation_methods or defaults["generation_methods"]
    if not set(methods) <= set(defaults["classification_methods"] + ["esm2_supervised"]):
        p.error("Unknown classification method")
    if not set(generators) <= {"true", "onehot", "rbm", "ankh"}:
        p.error("Quantitative generation supports only true, onehot, rbm, and ankh")
    stages = parse_stages(args.stages, profile["stages"])
    if "RR" in datasets and any(stage in stages for stage in (5, 6, 7, 8)) and 5000 not in budgets:
        p.error("Generation requires the 5,000-budget union for the True benchmark; include --train-sizes 5000")
    if args.profile in {"generation", "generation-pca"} and any(dataset != "RR" for dataset in datasets):
        p.error("Generation and generation analysis are configured for RR only")
    for dataset in datasets:
        if dataset not in config["datasets"]:
            p.error(f"Unknown dataset: {dataset}")
        ds = config["datasets"][dataset]
        t1_values = args.t1 or profile.get("t1", ds["t1"])
        seeds = args.seeds or profile.get("seeds", ds["seeds"])
        if not set(t1_values) <= set(ds["t1"]):
            p.error(f"Unsupported t1 for {dataset}: {t1_values}")
        case_stages = stages if ds["generation_enabled"] else [stage for stage in stages if stage not in (5, 6, 7, 8)]
        if len(case_stages) != len(stages):
            print(f"Skipping generation stages for {dataset}: RR only.")
        for t1 in t1_values:
            cases_ready = True
            for seed in seeds:
                cases_ready = run_case(dataset, t1, seed, ds, defaults, case_stages,
                                       methods, generators, budgets, args) is not False and cases_ready
            if cases_ready:
                run_analysis(dataset, t1, seeds, methods, generators, budgets, case_stages, args)
            elif args.plan:
                print(f"Skipping {dataset}/{t1} analysis plan: required case artifacts are blocked.")


def main() -> None:
    def interrupted(signum: int, _frame: object) -> None:
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    p = parser()
    args = p.parse_args()
    if args.jobs != 1:
        p.error("Parallel execution is not implemented; use --jobs 1")
    config = tomllib.loads(MANIFEST.read_text())
    args.outputs_root = args.outputs_root.resolve()
    args.models_root = args.models_root.resolve()
    if args.plan:
        execute(args, p, config)
        return

    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
    log_dir = args.outputs_root / "logs" / run_id
    log_dir.mkdir(parents=True, exist_ok=False)
    (log_dir / "resolved_config.toml").write_text(
        MANIFEST.read_text() + "\n[run_selection]\n" +
        f"profile = {json.dumps(args.profile)}\n" +
        f"datasets = {json.dumps(args.datasets or [])}\n" +
        f"t1 = {json.dumps(args.t1 or [])}\n" +
        f"seeds = {json.dumps(args.seeds or [])}\n" +
        f"train_sizes = {json.dumps(args.train_sizes or [])}\n"
    )
    original_stdout, original_stderr = sys.stdout, sys.stderr
    status = "failed"
    with (log_dir / "run.log").open("w") as log:
        sys.stdout = Tee(original_stdout, log)
        sys.stderr = Tee(original_stderr, log)
        try:
            print(f"Run log: {log_dir}")
            execute(args, p, config)
            status = "complete"
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout, sys.stderr = original_stdout, original_stderr
            (log_dir / "plan.tsv").write_text(
                "state\tstage\tartifact\n" +
                "".join(f"{state}\t{stage}\t{label}\n" for state, stage, label in PLAN_ROWS)
            )
            (log_dir / "status.json").write_text(json.dumps({"status": status, "profile": args.profile,
                                                               "run_id": run_id}, indent=2) + "\n")


if __name__ == "__main__":
    main()
