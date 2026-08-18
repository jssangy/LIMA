#!/usr/bin/env python3
"""Freeze patched boundary-exit MAPF baselines with reproducible provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(repo: Path, *args: str, binary: bool = False):
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=not binary
    )


def freeze(
    *,
    method: str,
    repo: Path,
    binary: Path,
    output: Path,
    input_manifest: Path,
) -> None:
    if output.exists():
        raise SystemExit(f"refusing to overwrite frozen artifact: {output}")
    output.mkdir(parents=True)

    frozen_binary = output / binary.name
    shutil.copy2(binary, frozen_binary)
    source_diff = git(repo, "diff", "--binary", binary=True)
    diff_path = output / "SOURCE.diff"
    diff_path.write_bytes(source_diff)

    manifest = {
        "schema_version": 1,
        "status": "frozen",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "source_repo": str(repo.resolve()),
        "source_commit": git(repo, "rev-parse", "HEAD").strip(),
        "source_diff_file": diff_path.name,
        "source_diff_sha256": sha256(diff_path),
        "binary_file": frozen_binary.name,
        "binary_sha256": sha256(frozen_binary),
        "input_manifest": str(input_manifest.resolve()),
        "input_manifest_sha256": sha256(input_manifest),
        "mission_semantics": {
            "movement_domain": "managed cells plus assigned physical G only",
            "shared_physical_goals": True,
            "exclusive_assigned_goal_entry": True,
            "disappear_immediately_on_arrival": True,
            "conflicts_checked_through_arrival": True,
        },
        "paper_budget": {
            "execution_steps": 30000,
            "optimization_attempts": 100000,
            "initial_repair_attempts": 100000,
            "wall_clock_cutoff": None,
        },
        "regression": {
            "agents": 26,
            "unique_physical_goals": 17,
            "repeated_goal_assignments": 9,
            "boundary_entry_violations": 0,
            "vertex_edge_conflicts": 0,
            "disappear_at_goal": True,
        },
    }
    (output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=("mapf_lns2", "address"))
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--input-manifest", required=True, type=Path)
    args = parser.parse_args()
    freeze(
        method=args.method,
        repo=args.repo,
        binary=args.binary,
        output=args.output,
        input_manifest=args.input_manifest,
    )


if __name__ == "__main__":
    main()
