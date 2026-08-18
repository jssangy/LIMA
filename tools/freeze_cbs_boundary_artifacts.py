#!/usr/bin/env python3
"""Freeze native CBS and its managed-boundary single-shot protocol."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "results/revision_final/frozen_artifacts_cbs_boundary_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8"
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def main() -> int:
    sources = {
        "lima": ROOT / "results/revision_final/frozen_artifacts_boundary_exit_v2/lima",
        "cbs_baseline": ROOT / "build_boundary_exit_v2/cbs_baseline",
        "cbs_baseline.cpp": ROOT / "app/cbs_baseline.cpp",
        "run_final_certified_oneshot.py": ROOT / "tools/run_final_certified_oneshot.py",
    }
    input_manifest = (
        ROOT / "results/revision_final/certified_inputs_boundary_exit_v2/MANIFEST.json"
    )
    missing = [str(path) for path in [*sources.values(), input_manifest] if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing freeze inputs: {missing}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, dict[str, str]] = {}
    for name, source in sources.items():
        destination = OUTPUT / name
        source_hash = sha256(source)
        if destination.exists() and sha256(destination) != source_hash:
            raise RuntimeError(f"refusing to overwrite changed frozen artifact: {destination}")
        if not destination.exists():
            shutil.copy2(source, destination)
        artifacts[name] = {
            "path": str(destination.relative_to(ROOT)),
            "sha256": source_hash,
        }

    manifest = {
        "schema_version": 1,
        "status": "frozen",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "artifacts": artifacts,
        "input": {
            "path": str(input_manifest.relative_to(ROOT)),
            "sha256": sha256(input_manifest),
        },
        "mission_contract": {
            "movement_domain": "managed-cell union plus the agent's assigned boundary G",
            "boundary_entry": "an active agent may enter only its assigned G",
            "completion": "the agent disappears immediately after occupying its assigned G on arrival",
            "repeated_physical_goals": True,
        },
        "protocol": {
            "method": "native CBS; never injected into LIMA",
            "densities": [1, 5, 10, 20, 30, 40, 50, 60, 65, 70],
            "scenarios": list(range(10)),
            "execution_horizon_steps": 30000,
            "high_level_expansion_limit": 100000,
            "wall_clock_cutoff": None,
            "early_stop": False,
        },
    }
    atomic_json(OUTPUT / "MANIFEST.json", manifest)
    print(OUTPUT / "MANIFEST.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
