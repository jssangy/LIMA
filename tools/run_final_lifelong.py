#!/usr/bin/env python3
"""Run fixed-sequence lifelong LIMA experiments with SWR and direct routing."""

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
    parser.add_argument(
        "--input-manifest", default="results/revision_final/lifelong_inputs_v2/MANIFEST.json")
    parser.add_argument(
        "--binary", default="results/revision_final/frozen_artifacts_step_v2/lima")
    parser.add_argument("--variants", default="swr,direct")
    parser.add_argument("--maps", help="optional comma-separated map filter")
    parser.add_argument("--densities", help="optional comma-separated density filter")
    parser.add_argument("--scenarios", help="optional comma-separated scenario filter")
    parser.add_argument("--horizon", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--output-dir", default="results/revision_final/lifelong_lima_step_v2")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    if not variants or not set(variants).issubset({"swr", "direct"}):
        parser.error("variants must be swr and/or direct")
    if args.jobs < 1 or args.horizon < 1 or args.warmup < 0 or args.warmup >= args.horizon:
        parser.error("invalid jobs, horizon, or warmup")
    map_filter = set(args.maps.split(",")) if args.maps else None
    density_filter = set(map(int, args.densities.split(","))) if args.densities else None
    scenario_filter = set(map(int, args.scenarios.split(","))) if args.scenarios else None

    binary = (ROOT / args.binary).resolve()
    if not binary.is_file():
        parser.error(f"missing lifelong binary: {binary}")
    version = subprocess.run(
        [str(binary), "--version"], cwd=ROOT, capture_output=True, text=True, check=False)
    if version.returncode != 0 or "profile=lima-default" not in version.stdout:
        parser.error("lifelong binary does not expose the frozen LIMA profile")
    input_manifest_path = (ROOT / args.input_manifest).resolve()
    inputs = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    if inputs.get("semantics", {}).get("sequence") != "fixed cyclic order; no consecutive duplicate":
        parser.error("input manifest is not the fixed cyclic lifelong dataset")

    cells: list[dict] = []
    for map_name, map_entry in inputs["maps"].items():
        if map_filter is not None and map_name not in map_filter:
            continue
        map_path = ROOT / map_entry["map_file"]
        if sha256(map_path) != map_entry["map_sha256"]:
            parser.error(f"map hash mismatch: {map_name}")
        for density_text, density_entry in map_entry["densities"].items():
            density = int(density_text)
            if density_filter is not None and density not in density_filter:
                continue
            agents = int(density_entry["agents"])
            for scenario_text, entry in density_entry["scenarios"].items():
                scenario = int(scenario_text)
                if scenario_filter is not None and scenario not in scenario_filter:
                    continue
                scenario_path = ROOT / entry["scenario_file"]
                sequence_path = ROOT / entry["sequence_file"]
                certificate_path = ROOT / entry["certificate_file"]
                for path, expected in (
                    (scenario_path, entry["scenario_sha256"]),
                    (sequence_path, entry["sequence_sha256"]),
                    (certificate_path, entry["certificate_sha256"]),
                ):
                    if sha256(path) != expected:
                        parser.error(f"lifelong input hash mismatch: {path}")
                for variant in variants:
                    cells.append({
                        "map": map_name, "density": density, "agents": agents,
                        "scenario": scenario, "variant": variant,
                        "map_file": map_entry["map_file"],
                        "scenario_file": entry["scenario_file"],
                        "sequence_file": entry["sequence_file"],
                        "certificate_file": entry["certificate_file"],
                        "tag": f"{map_name}_d{density:02d}_a{agents}_s{scenario}_{variant}",
                    })
    if not cells:
        parser.error("lifelong input filters selected no cells")

    output = (ROOT / args.output_dir).resolve()
    records, resources, logs, metrics = (
        output / "records", output / "resources", output / "logs", output / "metrics")
    for directory in (records, resources, logs, metrics):
        directory.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).resolve()
    fingerprint_payload = {
        "schema_version": 2, "algorithm": "lima",
        "semantic_scope": "lifelong; fixed cyclic per-agent goal sequences",
        "binary": str(binary.relative_to(ROOT)), "binary_sha256": sha256(binary),
        "binary_version": version.stdout.strip(), "runner_sha256": sha256(runner),
        "input_manifest_sha256": sha256(input_manifest_path),
        "variants": variants, "cells": [cell["tag"] for cell in cells],
        "horizon_steps": args.horizon, "warmup_steps": args.warmup,
        "termination_policy": "fixed discrete horizon; no wall-clock cutoff",
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
        "job_count": len(cells), "jobs_concurrency": args.jobs,
        "host": {"hostname": socket.gethostname(), "platform": platform.platform(),
                 "cpu_count": os.cpu_count()},
    })
    lock = output / ".RUNNING"
    if lock.exists():
        parser.error(f"campaign lock already exists: {lock}")
    lock.write_text(f"pid={os.getpid()}\nstarted_utc={datetime.now(timezone.utc).isoformat()}\n")

    def run(cell: dict) -> tuple[str, str]:
        tag = cell["tag"]
        record_path = records / f"{tag}.json"
        if record_path.exists() and not args.rerun:
            return tag, "skipped"
        resource_path = resources / f"{tag}.txt"
        log_path = logs / f"{tag}.log"
        resource_path.unlink(missing_ok=True)
        solver = [
            str(binary), "--profile", "lima-default", "--mode", "solve",
            "--map", cell["map_file"], "--scenario", cell["scenario_file"],
            "--agents", str(cell["agents"]), "--seed", str(cell["scenario"]),
            "--max-steps", str(args.horizon),
            "--stall-threshold", str(args.horizon + 1),
            "--goal-behavior", "lifelong",
            "--goal-sequences", cell["sequence_file"],
            "--routing", cell["variant"], "--no-trace", "--metrics", str(metrics / tag),
        ]
        command = [
            "/usr/bin/time", "-f",
            "max_rss_kb=%M\nuser_seconds=%U\nsystem_seconds=%S\nelapsed_seconds=%e",
            "-o", str(resource_path), *solver,
        ]
        started = time.time()
        proc = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, start_new_session=True)
        stdout, stderr = proc.communicate()
        returncode = proc.returncode
        result = parse_fields(stdout)
        horizon_completed = (
            returncode in (0, 2)
            and result.get("status") == "step_limit"
            and result.get("steps") == str(args.horizon)
        )
        telemetry = summarize_metrics(metrics / tag)
        validation_ok = telemetry["path_conformity"]["online_validation_ok"]
        log_path.write_text(stdout + ("\n[stderr]\n" + stderr if stderr else ""), encoding="utf-8")
        atomic_json(record_path, {
            **cell, "algorithm": "lima", "returncode": returncode,
            "timed_out": False,
            "horizon_completed": horizon_completed and validation_ok,
            "normal_termination": horizon_completed and validation_ok,
            "runner_wall_seconds": time.time() - started,
            "result": result, "resource": parse_resource(resource_path),
            "telemetry": telemetry,
            "metrics": str((metrics / tag).relative_to(ROOT)),
            "command": command, "log": str(log_path.relative_to(ROOT)),
            "experiment_fingerprint": fingerprint,
        })
        return tag, "horizon" if horizon_completed and validation_ok else "stopped"

    completed = 0
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
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
