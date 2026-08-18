#!/usr/bin/env python3
"""Freeze the managed-boundary stochastic and lifelong executables."""

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
OUTPUT = ROOT / "results/revision_final/frozen_artifacts_boundary_followups_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def atomic_json(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8"
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def main() -> int:
    pibt_repo = Path.home() / "mapf-baselines/pibt2"
    lacam_repo = Path.home() / "mapf-baselines/lacam"
    sources = {
        "lima": ROOT / "results/revision_final/frozen_artifacts_boundary_exit_v2/lima",
        "lacam": ROOT / "results/revision_final/frozen_artifacts_boundary_exit_v2/lacam",
        "pibt_native_stochastic": ROOT / "build_pibt_native_stochastic_boundary_exit_v2",
        "pibt_lifelong": pibt_repo / "build_boundary_exit_v2/lifelong_fixed",
        "pibt_native_stochastic.cpp": ROOT / "tools/pibt_native_stochastic.cpp",
        "pibt_lifelong.cpp": pibt_repo / "lifelong_fixed.cpp",
        "pibt_node.hpp": pibt_repo / "third_party/grid-pathfinding/graph/include/node.hpp",
        "pibt_node.cpp": pibt_repo / "third_party/grid-pathfinding/graph/src/node.cpp",
        "pibt_graph.cpp": pibt_repo / "third_party/grid-pathfinding/graph/src/graph.cpp",
        "stochastic_replan_adapter.py": ROOT / "tools/stochastic_replan_adapter.py",
        "pibt_native_stochastic_adapter.py": ROOT / "tools/pibt_native_stochastic_adapter.py",
        "run_final_stochastic.py": ROOT / "tools/run_final_stochastic.py",
        "run_final_lifelong.py": ROOT / "tools/run_final_lifelong.py",
        "run_bonus_lifelong_pibt.py": ROOT / "tools/run_bonus_lifelong_pibt.py",
        "generate_lifelong_sequences.py": ROOT / "tools/generate_lifelong_sequences.py",
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
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
    inputs = {
        "single_shot": ROOT / "results/revision_final/certified_inputs_boundary_exit_v2/MANIFEST.json",
        "lifelong": ROOT / "results/revision_final/lifelong_inputs_boundary_exit_v2/MANIFEST.json",
    }
    manifest = {
        "schema_version": 1,
        "status": "frozen",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commits": {
            "lima": commit(ROOT),
            "pibt2": commit(pibt_repo),
            "lacam": commit(lacam_repo),
        },
        "artifacts": artifacts,
        "inputs": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for name, path in inputs.items()
        },
        "mission_contract": {
            "movement_domain": "managed-cell union plus assigned boundary G",
            "boundary_entry": "only the active assigned G; departure from the just-completed G is transiently allowed in lifelong mode",
            "completion": "one-shot disappears on G arrival; lifelong installs the next cyclic boundary task in the same step",
            "repeated_physical_goals": True,
        },
        "protocol": {
            "stochastic": {
                "densities": [10, 20, 30],
                "probabilities": [0.05, 0.10, 0.15, 0.20],
                "seeds": [0, 1, 2, 3, 4],
                "horizon_steps": 30000,
            },
            "lifelong": {
                "densities": [10, 30, 50],
                "seeds": [0, 1, 2, 3, 4],
                "horizon_steps": 10000,
                "warmup_steps": 1000,
                "arms": ["lima+bfs", "lima+swr", "lima+static-guidance", "pibt"],
            },
        },
    }
    atomic_json(OUTPUT / "MANIFEST.json", manifest)
    print(OUTPUT / "MANIFEST.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
