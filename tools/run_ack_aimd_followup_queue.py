#!/usr/bin/env python3
"""Launch frozen LIMA-v3 follow-up campaigns after the core runs."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
REVISION = ROOT / "results/revision_final"
QUEUE = REVISION / "lima_v3_followup_queue"
CORE = {
    "oneshot_lima_v3_core": 210,
    "stochastic_lima_v3_liveness": 180,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_count(name: str) -> int:
    return len(list((REVISION / name / "records").glob("*.json")))


def lock_pid(name: str) -> int | None:
    lock = REVISION / name / ".RUNNING"
    if not lock.is_file():
        return None
    try:
        first_line = lock.read_text(encoding="utf-8").splitlines()[0]
        return int(first_line.split("=", 1)[1])
    except (IndexError, ValueError):
        return -1


def pid_alive(pid: int | None) -> bool:
    return pid is not None and pid > 0 and Path(f"/proc/{pid}").exists()


def wait_for_core() -> None:
    while True:
        ready = True
        status = []
        for name, expected in CORE.items():
            count = record_count(name)
            pid = lock_pid(name)
            alive = pid_alive(pid)
            status.append(f"{name}={count}/{expected},alive={alive}")
            if count != expected or alive:
                ready = False
            if not alive and count < expected:
                raise RuntimeError(
                    f"core campaign stopped early: {name} {count}/{expected}, pid={pid}")
        print(f"[{utc_now()}] " + "; ".join(status), flush=True)
        if ready:
            return
        time.sleep(60)


def validate_core_records() -> None:
    for name, expected in CORE.items():
        files = sorted((REVISION / name / "records").glob("*.json"))
        if len(files) != expected:
            raise RuntimeError(f"record count mismatch: {name} {len(files)}/{expected}")
        for path in files:
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
                    raise RuntimeError(f"{field} in {path}")


def launch(name: str, command: list[str]) -> tuple[subprocess.Popen, object]:
    QUEUE.mkdir(parents=True, exist_ok=True)
    log = (QUEUE / f"{name}.log").open("a", encoding="utf-8")
    log.write(f"\n[{utc_now()}] {' '.join(command)}\n")
    log.flush()
    process = subprocess.Popen(
        command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
        text=True, start_new_session=True,
    )
    return process, log


def main() -> int:
    QUEUE.mkdir(parents=True, exist_ok=True)
    wait_for_core()
    validate_core_records()

    frozen = "results/revision_final/frozen_artifacts_lima_v3"
    certified = "results/revision_final/certified_inputs_v3/MANIFEST.json"
    campaigns = {
        "oneshot_high": [
            "python3", "tools/run_final_certified_oneshot.py",
            "--algorithm", "lima", "--input-manifest", certified,
            "--maps", "cross_3030,warehouse_10_20,warehouse_20_40",
            "--targets", "d60,d65,d70,boundary",
            "--scenarios", "0,1,2,3,4,5,6,7,8,9",
            "--max-steps", "100000", "--lima-binary", f"{frozen}/lima",
            "--freeze-manifest", f"{frozen}/MANIFEST.json",
            "--no-early-stop", "--jobs", "4", "--output-dir",
            "results/revision_final/oneshot_lima_v3_high",
        ],
        "lifelong": [
            "python3", "tools/run_final_lifelong.py",
            "--input-manifest", "results/revision_final/lifelong_inputs_v2/MANIFEST.json",
            "--binary", f"{frozen}/lima", "--variants", "swr,direct",
            "--maps", "cross_3030,warehouse_10_20,warehouse_20_40",
            "--densities", "10,30,50", "--scenarios", "0,1,2,3,4",
            "--horizon", "10000", "--warmup", "1000", "--jobs", "6",
            "--output-dir", "results/revision_final/lifelong_lima_v3",
        ],
        "admission": [
            "python3", "tools/run_final_admission_ablation.py",
            "--freeze-manifest", f"{frozen}/MANIFEST.json",
            "--certified-manifest", certified, "--binary", f"{frozen}/lima",
            "--max-steps", "100000", "--jobs", "4", "--output-dir",
            "results/revision_final/admission_ablation_lima_v3",
        ],
        "local_solver": [
            "python3", "tools/run_final_local_solver.py",
            "--freeze-manifest", f"{frozen}/MANIFEST.json",
            "--instances", "100", "--seed", "1701", "--output-dir",
            "results/revision_final/local_solver_reference_lima_v3",
        ],
    }
    running = {name: launch(name, command) for name, command in campaigns.items()}
    failures = []
    for name, (process, log) in running.items():
        returncode = process.wait()
        log.write(f"[{utc_now()}] returncode={returncode}\n")
        log.close()
        if returncode != 0:
            failures.append((name, returncode))
    summary = {
        "completed_utc": utc_now(),
        "returncodes": {name: process.returncode for name, (process, _) in running.items()},
        "failures": failures,
    }
    (QUEUE / "STATUS.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError(f"follow-up failures: {failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
