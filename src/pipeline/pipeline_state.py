"""Artifact signatures and completion records for the consolidated runner."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, object]:
    """Compute a content checksum for an input or completed artifact."""
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size,
            "sha256": _digest(path.resolve())}


def signature(inputs: Iterable[Path], settings: dict[str, object]) -> str | None:
    paths = list(inputs)
    if any(not path.is_file() for path in paths):
        return None
    payload = {"inputs": [file_identity(path) for path in paths], "settings": settings}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def sidecar_path(anchor: Path) -> Path:
    return anchor.with_name(anchor.name + ".pipeline.json")


def inspect(outputs: Iterable[Path], anchor: Path, expected: str | None,
            allow_legacy: bool = False) -> str:
    paths = list(outputs)
    if expected is None:
        return "BLOCKED"
    if not paths or any(not path.is_file() or
                        (path.stat().st_size == 0 and not path.name.endswith('.complete'))
                        for path in paths):
        return "BUILD"
    sidecar = sidecar_path(anchor)
    if sidecar.is_file():
        try:
            recorded = json.loads(sidecar.read_text())
            if recorded.get("signature") != expected:
                return "STALE"
            identities = recorded.get("output_identities", [])
            if len(identities) != len(paths) or identities != [file_identity(path) for path in paths]:
                return "STALE"
            return "REUSE"
        except (OSError, ValueError):
            return "STALE"
    if allow_legacy:
        for path in paths:
            if path.suffix in {".h5", ".hdf5"}:
                try:
                    import h5py
                    with h5py.File(path, "r") as handle:
                        if not list(handle.keys()):
                            return "STALE"
                except (OSError, ValueError):
                    return "STALE"
        return "LEGACY_REUSE"
    return "STALE"


def record(anchor: Path, expected: str, outputs: Iterable[Path], legacy: bool = False,
           inputs: Iterable[Path] = (), settings: dict[str, object] | None = None) -> None:
    sidecar = sidecar_path(anchor)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    paths = list(outputs)
    payload = {"signature": expected, "outputs": [str(p) for p in paths],
               "output_identities": [file_identity(path) for path in paths],
               "input_identities": [file_identity(path) for path in inputs],
               "settings": settings or {}, "legacy_adopted": legacy,
               "created_at": datetime.now(timezone.utc).isoformat(),
               "status": "complete", "schema": 1}
    temp = sidecar.with_name(sidecar.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temp, sidecar)
