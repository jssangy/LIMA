#!/usr/bin/env python3
"""Start managed-boundary stochastic and lifelong campaigns after one-shot."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results/revision_final"
QUEUE = RESULTS / "queue_boundary_followups_v1"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8"
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def records(directory: Path) -> list[dict]:
    parsed = []
    for path in sorted((directory / "records").glob("*.json")):
        parsed.append(json.loads(path.read_text(encoding="utf-8")))
    return parsed


def one_shot_ready() -> tuple[bool, dict[str, object]]:
    campaigns = {
        "lima": (RESULTS / "oneshot_lima_managed_boundary_h30000_v2", 250),
        "lacam": (RESULTS / "oneshot_lacam_managed_boundary_h30000_v2", 250),
        "pibt": (RESULTS / "oneshot_pibt_managed_boundary_h30000_v2", 250),
    }
    summary: dict[str, object] = {}
    ready = True
    for name, (directory, expected) in campaigns.items():
        parsed = records(directory)
        bad_returncodes = sum(row.get("returncode", 0) not in (0, None) for row in parsed)
        locked = (directory / ".RUNNING").exists()
        summary[name] = {
            "records": len(parsed), "expected": expected,
            "bad_returncodes": bad_returncodes, "running_lock": locked,
            "solved": sum(bool(row.get("solved")) for row in parsed),
        }
        ready &= len(parsed) == expected and bad_returncodes == 0 and not locked
    return ready, summary


def command_specs() -> list[tuple[str, Path, list[str], int]]:
    freeze = RESULTS / "frozen_artifacts_boundary_followups_v1"
    common_stochastic = [
        "--input-manifest", "results/revision_final/certified_inputs_boundary_exit_v2/MANIFEST.json",
        "--maps", "cross_3030,warehouse_10_20,warehouse_20_40",
        "--densities", "10,20,30", "--scenarios", "0-4",
        "--probabilities", "0.05,0.10,0.15,0.20", "--max-steps", "30000",
        "--no-early-stop", "--trace-root",
        "results/revision_final/stochastic_trace_descriptors_boundary_exit_v2",
        "--lima", str(freeze / "lima"),
    ]
    return [
        ("stochastic_lima", RESULTS / "stochastic_lima_managed_boundary_h30000_v1", [
            sys.executable, "tools/run_final_stochastic.py", "--algorithm", "lima",
            *common_stochastic, "--jobs", "6", "--output-dir",
            "results/revision_final/stochastic_lima_managed_boundary_h30000_v1",
        ], 180),
        ("stochastic_lacam", RESULTS / "stochastic_lacam_replan_managed_boundary_h30000_v1", [
            sys.executable, "tools/run_final_stochastic.py", "--algorithm", "lacam-replan",
            *common_stochastic, "--jobs", "2", "--lacam-binary", str(freeze / "lacam"),
            "--output-dir", "results/revision_final/stochastic_lacam_replan_managed_boundary_h30000_v1",
        ], 180),
        ("stochastic_pibt", RESULTS / "stochastic_pibt_native_managed_boundary_h30000_v1", [
            sys.executable, "tools/run_final_stochastic.py", "--algorithm", "pibt-native",
            *common_stochastic, "--jobs", "4", "--pibt-native-binary",
            str(freeze / "pibt_native_stochastic"), "--output-dir",
            "results/revision_final/stochastic_pibt_native_managed_boundary_h30000_v1",
        ], 180),
        ("lifelong_lima", RESULTS / "lifelong_lima_managed_boundary_v2", [
            sys.executable, "tools/run_final_lifelong.py", "--input-manifest",
            "results/revision_final/lifelong_inputs_boundary_exit_v2/MANIFEST.json",
            "--binary", str(freeze / "lima"), "--variants", "bfs,swr,static-guidance",
            "--horizon", "10000", "--warmup", "1000", "--jobs", "3",
            "--movement-domain", "managed-boundary", "--output-dir",
            "results/revision_final/lifelong_lima_managed_boundary_v2",
        ], 135),
        ("lifelong_pibt", RESULTS / "lifelong_pibt_managed_boundary_v2", [
            sys.executable, "tools/run_bonus_lifelong_pibt.py", "--input-manifest",
            "results/revision_final/lifelong_inputs_boundary_exit_v2/MANIFEST.json",
            "--binary", str(freeze / "pibt_lifelong"), "--horizon", "10000",
            "--warmup", "1000", "--jobs", "2", "--movement-domain",
            "managed-boundary", "--output-dir",
            "results/revision_final/lifelong_pibt_managed_boundary_v2",
        ], 45),
    ]


def main() -> int:
    QUEUE.mkdir(parents=True, exist_ok=True)
    freeze = RESULTS / "frozen_artifacts_boundary_followups_v1/MANIFEST.json"
    if not freeze.is_file():
        raise FileNotFoundError(freeze)
    while True:
        ready, summary = one_shot_ready()
        atomic_json(QUEUE / "WAITING.json", {
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "ready": ready, "one_shot": summary,
        })
        print(datetime.now().isoformat(), "one-shot", summary, flush=True)
        if ready:
            break
        time.sleep(60)

    children: list[tuple[str, Path, int, subprocess.Popen[str], object]] = []
    for name, output, command, expected in command_specs():
        lock = output / ".RUNNING"
        if lock.exists():
            raise RuntimeError(f"refusing to overlap active campaign: {lock}")
        log_path = QUEUE / f"{name}.log"
        stream = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
            text=True, start_new_session=True,
        )
        children.append((name, output, expected, process, stream))
        print("started", name, "pid", process.pid, flush=True)

    while any(process.poll() is None for _, _, _, process, _ in children):
        status = {}
        for name, output, expected, process, _ in children:
            status[name] = {
                "pid": process.pid, "returncode": process.poll(),
                "records": len(records(output)), "expected": expected,
            }
        atomic_json(QUEUE / "STATUS.json", {
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "campaigns": status,
        })
        print(datetime.now().isoformat(), status, flush=True)
        time.sleep(60)

    failed = False
    final = {}
    for name, output, expected, process, stream in children:
        stream.close()
        parsed = records(output)
        bad = sum(row.get("returncode", 0) not in (0, None, 2) for row in parsed)
        final[name] = {
            "returncode": process.returncode, "records": len(parsed),
            "expected": expected, "bad_record_returncodes": bad,
        }
        failed |= process.returncode != 0 or len(parsed) != expected or bad != 0
    atomic_json(QUEUE / "FINAL.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "ok": not failed, "campaigns": final,
    })
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
