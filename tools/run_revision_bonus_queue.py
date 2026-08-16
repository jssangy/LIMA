#!/usr/bin/env python3
"""Start bonus comparisons only after every required revision campaign completes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
QUEUE_ROOT = ROOT / "results/revision_final/bonus_queue_v1"
REQUIRED = {
    "oneshot_lima_certified_step_v4_optimized": 280,
    "oneshot_primal2_certified_step_v6_common5000_stall256": 280,
    "oneshot_cbs_certified_step_v3": 280,
    "oneshot_lacam_certified_step_v3": 280,
    "oneshot_pibt_certified_step_v3": 280,
    "oneshot_cbs_telemetry_backfill_v1": 280,
    "oneshot_lacam_telemetry_backfill_v2": 280,
    "oneshot_pibt_telemetry_backfill_v2": 280,
    "stochastic_lima_step_v7_pgrid5_d10_30": 360,
    "stochastic_pibt_replan_step_v1_pgrid5_d10_30": 360,
    "stochastic_lacam_replan_step_v1_pgrid5_d10_30": 360,
    "lifelong_lima_step_v4_optimized": 180,
    "admission_ablation_step_v4_optimized": 24,
    "local_solver_reference_step_v4_optimized": 9,
}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8"
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def campaign_audit() -> dict[str, dict]:
    audit = {}
    base = ROOT / "results/revision_final"
    for name, expected in REQUIRED.items():
        root = base / name
        records = list((root / "records").glob("*.json"))
        malformed = 0
        unexpected = 0
        validation_errors = 0
        for path in records:
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                malformed += 1
                continue
            returncode = record.get("returncode")
            if returncode not in (0, None, 2):
                unexpected += 1
            if record.get("timed_out"):
                unexpected += 1
            if record.get("validation_error") or record.get("capacity_violation"):
                validation_errors += 1
            telemetry = record.get("telemetry") or {}
            conformity = telemetry.get("path_conformity") or {}
            backfill_conformity = (
                (record.get("trajectory") or {}).get("path_conformity") or {}
            )
            result = record.get("result") or {}
            if conformity.get("online_validation_ok") is False:
                validation_errors += 1
            if backfill_conformity.get("online_validation_ok") is False:
                validation_errors += 1
            equivalence = record.get("scalar_equivalence") or {}
            if equivalence and equivalence.get("all_match") is not True:
                validation_errors += 1
            for source in (conformity, backfill_conformity, result):
                for key in ("vertex_conflicts", "edge_conflicts", "invalid_moves"):
                    value = source.get(key)
                    if value not in (None, 0, "0"):
                        validation_errors += 1
        audit[name] = {
            "records": len(records), "expected": expected,
            "running": (root / ".RUNNING").exists(),
            "malformed": malformed, "unexpected": unexpected,
            "validation_errors": validation_errors,
            "ready": len(records) == expected and not (root / ".RUNNING").exists()
                     and malformed == 0 and unexpected == 0
                     and validation_errors == 0,
        }
    return audit


def run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"\n[{datetime.now(timezone.utc).isoformat()}] {json.dumps(command)}\n")
        stream.flush()
        process = subprocess.run(
            command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
            text=True, check=False)
        stream.write(f"[{datetime.now(timezone.utc).isoformat()}] returncode={process.returncode}\n")
    if process.returncode != 0:
        raise RuntimeError(f"command failed ({process.returncode}): {command[1]}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-root", default=str(QUEUE_ROOT))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--wait-for-required", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=900)
    args = parser.parse_args()
    if args.poll_seconds < 60:
        parser.error("--poll-seconds must be at least 60")
    queue_root = Path(args.queue_root).resolve()
    queue_root.mkdir(parents=True, exist_ok=True)
    while True:
        audit = campaign_audit()
        atomic_json(queue_root / "REQUIRED_AUDIT.json", audit)
        if all(row["ready"] for row in audit.values()):
            break
        if args.check_only:
            print(json.dumps(audit, indent=2, sort_keys=True))
            return 3
        if not args.wait_for_required:
            print("required campaigns are not complete; bonus queue not started", file=sys.stderr)
            return 3
        ready = sum(row["ready"] for row in audit.values())
        print(
            f"[{datetime.now(timezone.utc).isoformat()}] required campaigns "
            f"ready={ready}/{len(audit)}; checking again in {args.poll_seconds}s",
            flush=True)
        time.sleep(args.poll_seconds)
    if args.check_only:
        print(json.dumps(audit, indent=2, sort_keys=True))
        return 0

    lock = queue_root / ".RUNNING"
    if lock.exists():
        parser.error(f"bonus queue lock exists: {lock}")
    atomic_json(lock, {"pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat()})
    python = sys.executable
    commands = {
        "postprocess_required": [
            python, str(ROOT / "tools/summarize_paper_metrics.py"),
            "--output-dir", "results/revision_final/paper_metrics_v1"],
        "lifelong_required": [
            python, str(ROOT / "tools/summarize_lifelong.py"),
            str(ROOT / "results/revision_final/lifelong_lima_step_v4_optimized")],
        "smoke_lns2": [
            python, str(ROOT / "tools/run_bonus_mapf_lns2.py"),
            "--targets", "d01", "--scenarios", "0", "--jobs", "3",
            "--max-steps", "5000",
            "--output-dir", "results/revision_final/bonus_smoke_mapf_lns2_common5000_v2"],
        "smoke_pibt_lifelong": [
            python, str(ROOT / "tools/run_bonus_lifelong_pibt.py"),
            "--densities", "10", "--scenarios", "0",
            "--horizon", "100", "--warmup", "10", "--jobs", "3",
            "--output-dir", "results/revision_final/bonus_smoke_lifelong_pibt_v1"],
    }
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "required_campaigns": REQUIRED,
        "ordering": "required campaigns -> derived summaries -> adapter smoke -> bonus campaigns",
        "bonus": {
            "one_shot": "MAPF-LNS2, same 280 logical cells, 100000 high-level iterations and the common 5000-step execution horizon",
            "lifelong": "PIBT, same 90 fixed-sequence cells, 10000 steps with 1000-step warm-up",
            "agent_completion": "PIBT no-early-stop rerun over all 280 cells at the common 5000-step execution horizon; CBS/LaCAM intermediate states excluded",
        },
        "commands": commands,
    }
    atomic_json(queue_root / "MANIFEST.json", manifest)
    status: dict[str, str] = {}
    try:
        for name, command in commands.items():
            run(command, queue_root / "logs" / f"{name}.log")
            status[name] = "complete"
            atomic_json(queue_root / "STATUS.json", status)

        bonus_commands = {
            "mapf_lns2": [
                python, str(ROOT / "tools/run_bonus_mapf_lns2.py"), "--jobs", "5",
                "--max-steps", "5000",
                "--output-dir", "results/revision_final/bonus_mapf_lns2_common5000_v2"],
            "pibt_lifelong": [
                python, str(ROOT / "tools/run_bonus_lifelong_pibt.py"), "--jobs", "5",
                "--output-dir", "results/revision_final/bonus_lifelong_pibt_v1"],
            "pibt_agent_completion": [
                python, str(ROOT / "tools/run_final_certified_oneshot.py"),
                "--algorithm", "pibt", "--jobs", "5", "--no-early-stop",
                "--max-steps", "5000",
                "--freeze-manifest",
                "results/revision_final/frozen_artifacts_step_v4_optimized_r2/MANIFEST.json",
                "--lima-binary",
                "results/revision_final/frozen_artifacts_step_v4_optimized/lima",
                "--pibt-binary", "/home/shlee/mapf-baselines/pibt2/build_bonus/mapf",
                "--output-dir",
                "results/revision_final/bonus_pibt_agent_completion_common5000_v2"],
        }
        processes = {}
        streams = {}
        for name, command in bonus_commands.items():
            log = queue_root / "logs" / f"{name}.log"
            stream = log.open("a", encoding="utf-8")
            stream.write(f"\n[{datetime.now(timezone.utc).isoformat()}] {json.dumps(command)}\n")
            stream.flush()
            processes[name] = subprocess.Popen(
                command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                text=True, start_new_session=True)
            streams[name] = stream
            status[name] = "running"
        atomic_json(queue_root / "STATUS.json", status)
        failures = {}
        for name, process in processes.items():
            returncode = process.wait()
            streams[name].write(
                f"[{datetime.now(timezone.utc).isoformat()}] returncode={returncode}\n")
            streams[name].close()
            status[name] = "complete" if returncode == 0 else f"failed:{returncode}"
            if returncode != 0:
                failures[name] = returncode
            atomic_json(queue_root / "STATUS.json", status)
        if failures:
            raise RuntimeError(f"bonus campaign failures: {failures}")

        run([
            python, str(ROOT / "tools/summarize_paper_metrics.py"),
            "--campaign", "mapf-lns2=results/revision_final/bonus_mapf_lns2_common5000_v2",
            "--campaign", "pibt-acr=results/revision_final/bonus_pibt_agent_completion_common5000_v2",
            "--output-dir", "results/revision_final/paper_metrics_with_bonus_v1",
        ], queue_root / "logs/final_summaries.log")
        run([
            python, str(ROOT / "tools/summarize_lifelong.py"),
            str(ROOT / "results/revision_final/bonus_lifelong_pibt_v1"),
        ], queue_root / "logs/final_summaries.log")
        atomic_json(queue_root / "COMPLETE.json", {
            "completed_utc": datetime.now(timezone.utc).isoformat(), "status": status})
    finally:
        lock.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
