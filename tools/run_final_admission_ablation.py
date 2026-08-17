#!/usr/bin/env python3
"""Run the final AIMD-versus-static admission ablation.

All starts satisfy the operational local-capacity certificate.  Each map uses
one high-density rung and its certified capacity boundary.  Wall time is
descriptive only; the simulation terminates by the common discrete horizon.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from summarize_telemetry import summarize_metrics


ROOT = Path(__file__).resolve().parent.parent
FREEZE_MANIFEST = ROOT / "results/reference_instantiation_freeze_v1/FINAL_MANIFEST.json"
CERTIFIED_MANIFEST = ROOT / "results/revision_final/certified_inputs_v2/MANIFEST.json"
VARIANTS = {
    "aimd": (),
    "static": ("--gate-policy", "static"),
}
TARGETS = {
    "warehouse_10_20": ("d60", "boundary"),
    "warehouse_20_40": ("d50", "boundary"),
    "cross_3030": ("d70", "boundary"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as stream:
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
        r"^(\w+)=([^\n]+)$", path.read_text(encoding="utf-8", errors="replace"), re.MULTILINE))


def load_cells(certified_path: Path) -> tuple[list[dict], dict]:
    cells: list[dict] = []
    certified = json.loads(certified_path.read_text(encoding="utf-8"))
    if certified.get("capacity_formula") != "sum(arm capacities) - longest arm":
        raise ValueError("certified manifest does not use operational capacity")
    for map_name, targets in TARGETS.items():
        map_entry = certified["maps"][map_name]
        if sha256(ROOT / map_entry["map_file"]) != map_entry["map_sha256"]:
            raise ValueError(f"map hash mismatch: {map_name}")
        for target in targets:
            target_entry = map_entry["targets"][target]
            for scenario in (0, 1):
                entry = target_entry["scenarios"][str(scenario)]
                scenario_path = ROOT / entry["scenario_file"]
                certificate_path = ROOT / entry["certificate_file"]
                if sha256(scenario_path) != entry["scenario_sha256"]:
                    raise ValueError(f"scenario hash mismatch: {scenario_path}")
                if sha256(certificate_path) != entry["certificate_sha256"]:
                    raise ValueError(f"certificate hash mismatch: {certificate_path}")
                certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
                if certificate["validation"]["capacity_violations"] != 0:
                    raise ValueError(f"capacity certificate failed: {certificate_path}")
                agents = int(target_entry["agents"])
                cells.append({
                    "scope": "capacity_certified_high_density",
                    "map": map_name,
                    "target": target,
                    "agents": agents,
                    "tile_density_percent": target_entry["tile_density_percent"],
                    "capacity_load_percent": target_entry["capacity_load_percent"],
                    "scenario": scenario,
                    "map_file": map_entry["map_file"],
                    "scenario_file": entry["scenario_file"],
                    "certificate_file": entry["certificate_file"],
                    "tag": f"certified_{map_name}_{target}_a{agents}_s{scenario}",
                })
    return cells, certified


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-manifest", default=str(FREEZE_MANIFEST))
    parser.add_argument("--certified-manifest", default=str(CERTIFIED_MANIFEST))
    parser.add_argument(
        "--binary", default="results/revision_final/frozen_artifacts_step_v2/lima")
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--output-dir", default="results/revision_final/admission_ablation_step_v2")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1 or args.max_steps < 1:
        parser.error("jobs and max-steps must be positive")

    freeze_path = Path(args.freeze_manifest).resolve()
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "frozen":
        parser.error("freeze manifest is not frozen")
    artifact = (
        freeze["artifacts"].get("lima_binary")
        or freeze["artifacts"]["lima"]
    )
    binary = (ROOT / args.binary).resolve()
    if not binary.is_file():
        parser.error("instrumented LIMA binary is missing")
    version = subprocess.run(
        [str(binary), "--version"], cwd=ROOT,
        capture_output=True, text=True, check=False)
    if version.returncode != 0 or "profile=lima-default" not in version.stdout:
        parser.error("instrumented LIMA binary does not expose the frozen profile")
    certified_path = Path(args.certified_manifest).resolve()
    try:
        cells, _ = load_cells(certified_path)
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    for cell in cells:
        if not (ROOT / cell["map_file"]).is_file() or not (ROOT / cell["scenario_file"]).is_file():
            parser.error(f"missing input for {cell['tag']}")

    output = (ROOT / args.output_dir).resolve()
    records, resources, logs, metrics = (
        output / "records", output / "resources", output / "logs", output / "metrics")
    for directory in (records, resources, logs, metrics):
        directory.mkdir(parents=True, exist_ok=True)

    runner = Path(__file__).resolve()
    jobs = [(cell, variant) for cell in cells for variant in VARIANTS]
    fingerprint_payload = {
        "schema_version": 2,
        "semantic_scope": "one-shot; AIMD versus static admission; disappear at physical sink",
        "freeze_commit": (
            freeze.get("git_commit")
            or freeze.get("source_commit")
            or freeze["protocol_commit"]
        ),
        "reference_lima_sha256": artifact["sha256"],
        "binary_sha256": sha256(binary),
        "binary_version": version.stdout.strip(),
        "runner_sha256": sha256(runner),
        "certified_manifest_sha256": sha256(certified_path),
        "cells": [cell["tag"] for cell in cells],
        "variants": VARIANTS,
        "termination_policy": "common discrete execution horizon; no wall-clock cutoff",
        "max_steps": args.max_steps,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")).hexdigest()
    manifest_path = output / "MANIFEST.json"
    if manifest_path.is_file() and any(records.glob("*.json")):
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("experiment_fingerprint") != fingerprint:
            parser.error("output directory contains a different experiment fingerprint")
    atomic_json(manifest_path, {
        **fingerprint_payload,
        "experiment_fingerprint": fingerprint,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runner": str(runner.relative_to(ROOT)),
        "job_count": len(jobs),
        "jobs_concurrency": args.jobs,
        "host": {"hostname": socket.gethostname(), "platform": platform.platform(),
                 "cpu_count": os.cpu_count()},
    })

    lock = output / ".RUNNING"
    if lock.exists():
        parser.error(f"campaign lock already exists: {lock}")
    lock.write_text(f"pid={os.getpid()}\nstarted_utc={datetime.now(timezone.utc).isoformat()}\n")

    def run(job: tuple[dict, str]) -> tuple[str, str]:
        cell, variant = job
        tag = f"{cell['tag']}_{variant}"
        record_path = records / f"{tag}.json"
        if record_path.exists() and not args.rerun:
            return tag, "skipped"
        resource_path = resources / f"{tag}.txt"
        log_path = logs / f"{tag}.log"
        metric_path = metrics / tag
        resource_path.unlink(missing_ok=True)
        solver = [
            str(binary), "--profile", "lima-default", "--mode", "solve",
            "--map", cell["map_file"], "--scenario", cell["scenario_file"],
            "--agents", str(cell["agents"]), "--seed", str(cell["scenario"]),
            "--max-steps", str(args.max_steps),
            "--stall-threshold", str(args.max_steps + 1),
            "--goal-behavior", "disappear",
            "--no-trace", "--metrics", str(metric_path),
            *VARIANTS[variant],
        ]
        command = [
            "/usr/bin/time", "-f",
            "max_rss_kb=%M\nuser_seconds=%U\nsystem_seconds=%S\nelapsed_seconds=%e",
            "-o", str(resource_path), *solver,
        ]
        started = time.time()
        proc = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, start_new_session=True)
        stdout, stderr = proc.communicate()
        returncode = proc.returncode
        result = parse_fields(stdout)
        telemetry = summarize_metrics(metric_path)
        solved = (
            returncode == 0
            and result.get("status") == "completed"
            and telemetry["path_conformity"]["online_validation_ok"]
        )
        log_path.write_text(stdout + ("\n[stderr]\n" + stderr if stderr else ""), encoding="utf-8")
        atomic_json(record_path, {
            **cell, "variant": variant, "returncode": returncode, "timed_out": False,
            "solved": solved, "runner_wall_seconds": time.time() - started,
            "result": result, "resource": parse_resource(resource_path),
            "telemetry": telemetry,
            "command": command, "log": str(log_path.relative_to(ROOT)),
            "metrics": str(metric_path.relative_to(ROOT)),
            "experiment_fingerprint": fingerprint,
        })
        return tag, "completed" if solved else result.get("status", f"rc{returncode}")

    completed = 0
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(run, job) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                tag, status = future.result()
                completed += 1
                print(f"[{completed:2d}/{len(jobs):2d}] {status:10s} {tag}", flush=True)
    finally:
        lock.unlink(missing_ok=True)
    print(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
