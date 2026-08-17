#!/usr/bin/env python3
"""Run deferred high-density LIMA-v3 cells after the selected core campaigns."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
REVISION = ROOT / "results/revision_final"
QUEUE = REVISION / "lima_v3_selective_followup"
REQUIRED = {
    "oneshot_lima_v3_known_straggler": 1,
    "stochastic_lima_v3_impacted": 20,
    "lifelong_lima_v3": 90,
    "admission_ablation_lima_v3": 24,
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def records(name: str) -> list[Path]:
    return sorted((REVISION / name / "records").glob("*.json"))


def running(name: str) -> bool:
    lock = REVISION / name / ".RUNNING"
    if not lock.is_file():
        return False
    try:
        pid = int(lock.read_text(encoding="utf-8").splitlines()[0].split("=", 1)[1])
    except (IndexError, ValueError):
        return False
    return pid > 0 and Path(f"/proc/{pid}").exists()


def validate(name: str, expected: int) -> None:
    paths = records(name)
    if len(paths) != expected:
        raise RuntimeError(f"{name}: {len(paths)}/{expected} records")
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("returncode") not in (0, 2):
            raise RuntimeError(f"unexpected returncode: {path}")
        conformity = record.get("telemetry", {}).get("path_conformity", {})
        if conformity.get("online_validation_ok") is not True:
            raise RuntimeError(f"path validation failure: {path}")
        for field in (
            "invalid_moves", "vertex_conflicts", "edge_conflicts",
            "completed_goal_mismatches", "goal_preservation_failures",
        ):
            if conformity.get(field, 0) != 0:
                raise RuntimeError(f"{field}: {path}")


def main() -> int:
    QUEUE.mkdir(parents=True, exist_ok=True)
    while True:
        state = []
        ready = True
        for name, expected in REQUIRED.items():
            count = len(records(name))
            alive = running(name)
            state.append(f"{name}={count}/{expected},alive={alive}")
            if count != expected or alive:
                ready = False
            if count < expected and not alive:
                raise RuntimeError(f"campaign stopped early: {name} {count}/{expected}")
        print(f"[{now()}] " + "; ".join(state), flush=True)
        if ready:
            break
        time.sleep(60)

    for name, expected in REQUIRED.items():
        validate(name, expected)

    frozen = "results/revision_final/frozen_artifacts_lima_v3"
    command = [
        "python3", "tools/run_final_certified_oneshot.py",
        "--algorithm", "lima",
        "--input-manifest", "results/revision_final/certified_inputs_v3/MANIFEST.json",
        "--maps", "cross_3030,warehouse_10_20,warehouse_20_40",
        "--targets", "d60,d65,d70",
        "--scenarios", "0,1,2,3,4,5,6,7,8,9",
        "--max-steps", "100000",
        "--lima-binary", f"{frozen}/lima",
        "--freeze-manifest", f"{frozen}/MANIFEST.json",
        "--no-early-stop", "--jobs", "8",
        "--output-dir", "results/revision_final/oneshot_lima_v3_high_deferred",
    ]
    log_path = QUEUE / "high_density.log"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{now()}] {' '.join(command)}\n")
        log.flush()
        completed = subprocess.run(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
    status = {
        "completed_utc": now(),
        "high_density_returncode": completed.returncode,
        "high_density_records": len(records("oneshot_lima_v3_high_deferred")),
    }
    (QUEUE / "STATUS.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
