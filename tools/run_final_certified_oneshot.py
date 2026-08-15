#!/usr/bin/env python3
"""Run LIMA, CBS, or PRIMAL2 on capacity-certified one-shot inputs."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import re
import signal
import socket
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
FREEZE_MANIFEST = ROOT / "results/reference_instantiation_freeze_v1/FINAL_MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8"
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def parse_fields(text: str) -> dict[str, str]:
    lines = [line for line in text.splitlines() if line.strip()]
    return dict(re.findall(r"(\w+)=([^\s]+)", lines[-1])) if lines else {}


def parse_resource(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    return dict(re.findall(
        r"^(\w+)=([^\n]+)$",
        path.read_text(encoding="utf-8", errors="replace"), re.MULTILINE))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", required=True, choices=("lima", "cbs", "primal2"))
    parser.add_argument(
        "--input-manifest",
        default="results/revision_final/certified_inputs_v1/MANIFEST.json")
    parser.add_argument("--maps", help="optional comma-separated map filter")
    parser.add_argument("--targets", help="optional comma-separated target-label filter")
    parser.add_argument("--scenarios", help="optional comma-separated scenario-index filter")
    parser.add_argument("--wall-budget", type=float, default=300.0)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--jobs", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--freeze-manifest", default=str(FREEZE_MANIFEST))
    parser.add_argument(
        "--primal-python", default=str(Path.home() / "miniconda3/envs/primal2/bin/python"))
    parser.add_argument(
        "--primal-script", default=str(Path.home() / "mapf-baselines/PRIMAL2/run_our_instances.py"))
    args = parser.parse_args()
    jobs_concurrency = args.jobs or (8 if args.algorithm == "lima" else 2)
    if jobs_concurrency < 1 or args.wall_budget <= 0 or args.max_steps < 1:
        parser.error("jobs, wall-budget, and max-steps must be positive")

    freeze_path = Path(args.freeze_manifest).resolve()
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "frozen":
        parser.error("freeze manifest is not frozen")
    lima_artifact = freeze["artifacts"]["lima_binary"]
    cbs_artifact = freeze["artifacts"]["cbs_binary"]
    lima = (ROOT / lima_artifact["path"]).resolve()
    cbs = (ROOT / cbs_artifact["path"]).resolve()
    primal_python = Path(args.primal_python).resolve()
    primal_script = Path(args.primal_script).resolve()
    executables = {
        "lima": [lima], "cbs": [cbs], "primal2": [primal_python, primal_script]
    }[args.algorithm]
    for path in executables:
        if not path.is_file():
            parser.error(f"missing executable or adapter: {path}")
    if sha256(lima) != lima_artifact["sha256"] or sha256(cbs) != cbs_artifact["sha256"]:
        parser.error("frozen LIMA or CBS binary hash mismatch")

    input_manifest_path = (ROOT / args.input_manifest).resolve()
    inputs = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    if inputs.get("capacity_formula") != "sum(arm capacities) - longest arm":
        parser.error("certified input manifest does not use the operational capacity")
    map_filter = set(args.maps.split(",")) if args.maps else None
    target_filter = set(args.targets.split(",")) if args.targets else None
    scenario_filter = set(map(int, args.scenarios.split(","))) if args.scenarios else None
    cells: list[dict] = []
    for map_name, map_entry in inputs["maps"].items():
        if map_filter is not None and map_name not in map_filter:
            continue
        map_path = ROOT / map_entry["map_file"]
        if sha256(map_path) != map_entry["map_sha256"]:
            parser.error(f"map hash mismatch: {map_name}")
        for target, target_entry in map_entry["targets"].items():
            if target_filter is not None and target not in target_filter:
                continue
            agents = int(target_entry["agents"])
            for scenario_text, entry in target_entry["scenarios"].items():
                scenario = int(scenario_text)
                if scenario_filter is not None and scenario not in scenario_filter:
                    continue
                scenario_path = ROOT / entry["scenario_file"]
                certificate_path = ROOT / entry["certificate_file"]
                if sha256(scenario_path) != entry["scenario_sha256"]:
                    parser.error(f"scenario hash mismatch: {scenario_path}")
                if sha256(certificate_path) != entry["certificate_sha256"]:
                    parser.error(f"certificate hash mismatch: {certificate_path}")
                certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
                if certificate["validation"]["capacity_violations"] != 0:
                    parser.error(f"capacity certificate failed: {certificate_path}")
                cells.append({
                    "map": map_name, "target": target, "agents": agents,
                    "scenario": scenario, "map_file": map_entry["map_file"],
                    "scenario_file": entry["scenario_file"],
                    "scenario_sha256": entry["scenario_sha256"],
                    "certificate_file": entry["certificate_file"],
                    "certificate_sha256": entry["certificate_sha256"],
                    "tag": f"{map_name}_{target}_a{agents}_s{scenario}",
                })
    if not cells:
        parser.error("certified input filters selected no cells")

    output = (ROOT / (args.output_dir or
        f"results/revision_final/oneshot_{args.algorithm}_certified_v1")).resolve()
    records, resources, logs, metrics = (
        output / "records", output / "resources", output / "logs", output / "metrics")
    for directory in (records, resources, logs, metrics):
        directory.mkdir(parents=True, exist_ok=True)

    runner = Path(__file__).resolve()
    fingerprint_payload = {
        "schema_version": 1,
        "semantic_scope": "one-shot; capacity-certified starts; disappear at physical sink",
        "algorithm": args.algorithm,
        "executables": {str(path): sha256(path) for path in executables},
        "runner_sha256": sha256(runner),
        "input_manifest_sha256": sha256(input_manifest_path),
        "cells": [cell["tag"] for cell in cells],
        "wall_budget_seconds": args.wall_budget, "max_steps": args.max_steps,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")).hexdigest()
    manifest_path = output / "MANIFEST.json"
    if manifest_path.is_file() and any(records.glob("*.json")):
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("experiment_fingerprint") != fingerprint:
            parser.error("output directory contains a different experiment fingerprint")
    atomic_json(manifest_path, {
        **fingerprint_payload, "experiment_fingerprint": fingerprint,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runner": str(runner.relative_to(ROOT)),
        "input_manifest": str(input_manifest_path.relative_to(ROOT)),
        "freeze_manifest": str(freeze_path.relative_to(ROOT)),
        "job_count": len(cells), "jobs_concurrency": jobs_concurrency,
        "host": {"hostname": socket.gethostname(), "platform": platform.platform(),
                 "cpu_count": os.cpu_count()},
    })
    lock = output / ".RUNNING"
    if lock.exists():
        parser.error(f"campaign lock already exists: {lock}")
    lock.write_text(f"pid={os.getpid()}\nstarted_utc={datetime.now(timezone.utc).isoformat()}\n")

    def run(cell: dict) -> tuple[str, str]:
        tag = cell["tag"]
        record_path = records / f"{tag}_{args.algorithm}.json"
        if record_path.exists() and not args.rerun:
            return tag, "skipped"
        resource_path = resources / f"{tag}_{args.algorithm}.txt"
        log_path = logs / f"{tag}_{args.algorithm}.log"
        resource_path.unlink(missing_ok=True)
        if args.algorithm == "lima":
            solver = [
                str(lima), "--profile", "lima-default", "--mode", "solve",
                "--map", cell["map_file"], "--scenario", cell["scenario_file"],
                "--agents", str(cell["agents"]), "--seed", str(cell["scenario"]),
                "--max-steps", str(args.max_steps), "--no-trace",
                "--metrics", str(metrics / tag),
            ]
        elif args.algorithm == "cbs":
            solver = [
                str(cbs), "--map", cell["map_file"], "--scenario", cell["scenario_file"],
                "--agents", str(cell["agents"]), "--time-limit", str(args.wall_budget),
            ]
        else:
            solver = [
                str(primal_python), str(primal_script),
                "--map", str((ROOT / cell["map_file"]).resolve()),
                "--scen", str((ROOT / cell["scenario_file"]).resolve()),
                "-n", str(cell["agents"]), "--seed", str(1234 + cell["scenario"]),
                "--max-steps", str(args.max_steps), "--progress-every", "0",
            ]
        command = [
            "/usr/bin/time", "-f",
            "max_rss_kb=%M\nuser_seconds=%U\nsystem_seconds=%S\nelapsed_seconds=%e",
            "-o", str(resource_path), *solver,
        ]
        started = time.time()
        timed_out = False
        proc = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, start_new_session=True)
        try:
            stdout, stderr = proc.communicate(timeout=args.wall_budget + 5)
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                stdout, stderr = proc.communicate()
            returncode = 124
        result = parse_fields(stdout)
        if args.algorithm == "primal2" and "completed" in result:
            done, total = result["completed"].split("/", 1)
            result["solved"] = "1" if done == total else "0"
            result["makespan"] = result.get("steps", "")
            result["elapsed_s"] = result.get("wall_s", "")
        solved = (
            result.get("status") == "completed" if args.algorithm == "lima"
            else result.get("solved") == "1")
        log_path.write_text(stdout + ("\n[stderr]\n" + stderr if stderr else ""), encoding="utf-8")
        atomic_json(record_path, {
            **cell, "algorithm": args.algorithm, "returncode": returncode,
            "timed_out": timed_out, "solved": solved,
            "runner_wall_seconds": time.time() - started,
            "result": result, "resource": parse_resource(resource_path),
            "command": command, "log": str(log_path.relative_to(ROOT)),
            "experiment_fingerprint": fingerprint,
        })
        return tag, "timeout" if timed_out else ("solved" if solved else "unsolved")

    completed = 0
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs_concurrency) as pool:
            futures = [pool.submit(run, cell) for cell in cells]
            for future in concurrent.futures.as_completed(futures):
                tag, status = future.result()
                completed += 1
                print(f"[{completed:3d}/{len(cells):3d}] {status:8s} {tag}", flush=True)
    finally:
        lock.unlink(missing_ok=True)
    print(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
