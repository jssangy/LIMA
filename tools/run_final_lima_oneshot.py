#!/usr/bin/env python3
"""Run the final LIMA deterministic one-shot benchmark.

This runner consumes the already-frozen binary artifact instead of rebuilding
it at a tools-only git commit. The binary and reference-config SHA-256 values
are checked against the canonical freeze manifest before any job is admitted.
Each cell writes an atomic JSON record and an independent resource record.
"""

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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FREEZE_MANIFEST = ROOT / "results/reference_instantiation_freeze_v1/FINAL_MANIFEST.json"
DENSITIES = (1, 5, 10, 20, 30, 40, 50, 60, 65, 70, 75)


@dataclass(frozen=True)
class Instance:
    map_file: str
    scenario_template: str
    tiles: int


INSTANCES = {
    "warehouse_10_20": Instance(
        "data/maps/warehouse_10_20_paper.map",
        "data/scenarios/warehouse-10-20-paper/warehouse-10-20-paper_s{s}.scen",
        2649,
    ),
    "warehouse_20_40": Instance(
        "data/maps/warehouse_20_40_paper.map",
        "data/scenarios/warehouse-20-40-paper/warehouse-20-40-paper_s{s}.scen",
        10499,
    ),
    "cross_3030": Instance(
        "data/maps/cross_3030_paper.map",
        "data/scenarios/cross-30-30-paper/cross-30-30-paper_s{s}.scen",
        10200,
    ),
}


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


def parse_int_list(text: str, allowed: set[int]) -> list[int]:
    values: set[int] = set()
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            lower, upper = (int(value) for value in item.split("-", 1))
            values.update(range(lower, upper + 1))
        else:
            values.add(int(item))
    if not values or not values.issubset(allowed):
        raise argparse.ArgumentTypeError(f"values must be a nonempty subset of {sorted(allowed)}")
    return sorted(values)


def parse_summary(text: str) -> dict[str, str]:
    lines = [line for line in text.splitlines() if line.strip()]
    return dict(re.findall(r"(\w+)=([^\s]+)", lines[-1])) if lines else {}


def parse_resource(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    return dict(
        re.findall(
            r"^(\w+)=([^\n]+)$",
            path.read_text(encoding="utf-8", errors="replace"),
            re.MULTILINE,
        )
    )


def validate_frozen_artifacts(
    manifest_path: Path, binary_override: str | None
) -> tuple[dict, Path]:
    freeze = json.loads(manifest_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "frozen":
        raise ValueError(f"freeze manifest is not frozen: {manifest_path}")
    artifact = freeze["artifacts"]["lima_binary"]
    binary = (ROOT / (binary_override or artifact["path"])).resolve()
    if not binary.is_file():
        raise FileNotFoundError(f"missing frozen binary: {binary}")
    if binary_override is None and sha256(binary) != artifact["sha256"]:
        raise ValueError("frozen LIMA binary SHA-256 does not match FINAL_MANIFEST.json")
    config_artifact = freeze["artifacts"]["reference_config"]
    config_path = (ROOT / config_artifact["path"]).resolve()
    if sha256(config_path) != config_artifact["sha256"]:
        raise ValueError("reference config SHA-256 does not match FINAL_MANIFEST.json")
    return freeze, binary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-manifest", default=str(DEFAULT_FREEZE_MANIFEST))
    parser.add_argument("--binary", help="exploratory override; disables manifest binary-hash assertion")
    parser.add_argument("--maps", default=",".join(INSTANCES))
    parser.add_argument("--densities", default=",".join(map(str, DENSITIES)))
    parser.add_argument("--scenarios", default="0-9")
    parser.add_argument("--wall-budget", type=float, default=300.0)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--output-dir", default="results/revision_final/oneshot_lima_standard_v1")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    if args.jobs < 1 or args.wall_budget <= 0 or args.max_steps < 1:
        parser.error("jobs, wall-budget, and max-steps must be positive")
    maps = [item.strip() for item in args.maps.split(",") if item.strip()]
    if not maps or not set(maps).issubset(INSTANCES):
        parser.error("unknown or empty map selection")
    densities = parse_int_list(args.densities, set(DENSITIES))
    scenarios = parse_int_list(args.scenarios, set(range(10)))
    freeze_manifest = Path(args.freeze_manifest).resolve()
    try:
        freeze, binary = validate_frozen_artifacts(freeze_manifest, args.binary)
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    output = (ROOT / args.output_dir).resolve()
    records = output / "records"
    resources = output / "resources"
    logs = output / "logs"
    metrics = output / "metrics"
    for directory in (records, resources, logs, metrics):
        directory.mkdir(parents=True, exist_ok=True)

    jobs: list[dict] = []
    inputs: dict[str, dict[str, object]] = {}
    for map_name in maps:
        spec = INSTANCES[map_name]
        map_path = ROOT / spec.map_file
        if not map_path.is_file():
            parser.error(f"missing map: {map_path}")
        inputs[spec.map_file] = {"sha256": sha256(map_path), "size_bytes": map_path.stat().st_size}
        for scenario in scenarios:
            scenario_file = spec.scenario_template.format(s=scenario)
            scenario_path = ROOT / scenario_file
            if not scenario_path.is_file():
                parser.error(f"missing scenario: {scenario_path}")
            inputs[scenario_file] = {
                "sha256": sha256(scenario_path),
                "size_bytes": scenario_path.stat().st_size,
            }
        for density in densities:
            agents = density * spec.tiles // 100
            for scenario in scenarios:
                jobs.append(
                    {
                        "map": map_name,
                        "density": density,
                        "agents": agents,
                        "scenario": scenario,
                        "map_file": spec.map_file,
                        "scenario_file": spec.scenario_template.format(s=scenario),
                        "tag": f"{map_name}_d{density:02d}_a{agents}_s{scenario}",
                    }
                )

    runner_path = Path(__file__).resolve()
    fingerprint_payload = {
        "schema_version": 1,
        "semantic_scope": "one-shot; disappear at physical sink",
        "profile": "lima-default",
        "freeze_commit": freeze["git_commit"],
        "binary_sha256": sha256(binary),
        "runner_sha256": sha256(runner_path),
        "maps": maps,
        "densities": densities,
        "scenarios": scenarios,
        "wall_budget_seconds": args.wall_budget,
        "max_steps": args.max_steps,
        "inputs": {path: item["sha256"] for path, item in sorted(inputs.items())},
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest_path = output / "MANIFEST.json"
    if manifest_path.is_file() and any(records.glob("*.json")):
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("experiment_fingerprint") != fingerprint:
            parser.error("output directory contains a different experiment fingerprint")

    manifest = {
        **fingerprint_payload,
        "experiment_fingerprint": fingerprint,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_manifest": str(freeze_manifest.relative_to(ROOT)),
        "binary": str(binary.relative_to(ROOT)),
        "runner": str(runner_path.relative_to(ROOT)),
        "job_count": len(jobs),
        "jobs_concurrency": args.jobs,
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
        "input_files": inputs,
    }
    atomic_json(manifest_path, manifest)

    running_lock = output / ".RUNNING"
    if running_lock.exists():
        parser.error(f"campaign lock already exists: {running_lock}")
    running_lock.write_text(
        f"pid={os.getpid()}\nstarted_utc={datetime.now(timezone.utc).isoformat()}\n"
    )

    def run(job: dict) -> tuple[str, str]:
        tag = job["tag"]
        record_path = records / f"{tag}.json"
        if record_path.exists() and not args.rerun:
            return tag, "skipped"
        resource_path = resources / f"{tag}.txt"
        log_path = logs / f"{tag}.log"
        resource_path.unlink(missing_ok=True)
        solver_command = [
            str(binary),
            "--profile", "lima-default",
            "--mode", "solve",
            "--map", job["map_file"],
            "--scenario", job["scenario_file"],
            "--agents", str(job["agents"]),
            "--seed", str(job["scenario"]),
            "--max-steps", str(args.max_steps),
            "--no-trace",
            "--metrics", str(metrics / tag),
        ]
        command = [
            "/usr/bin/time", "-f",
            "max_rss_kb=%M\nuser_seconds=%U\nsystem_seconds=%S\nelapsed_seconds=%e",
            "-o", str(resource_path),
            *solver_command,
        ]
        started = time.time()
        timed_out = False
        proc = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=args.wall_budget)
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
        wall_seconds = time.time() - started
        log_path.write_text(
            stdout + ("\n[stderr]\n" + stderr if stderr else ""), encoding="utf-8"
        )
        summary = parse_summary(stdout)
        payload = {
            **job,
            "algorithm": "lima",
            "profile": "lima-default",
            "semantic_scope": "one-shot; disappear at physical sink",
            "returncode": returncode,
            "timed_out": timed_out,
            "runner_wall_seconds": wall_seconds,
            "summary": summary,
            "resource": parse_resource(resource_path),
            "command": command,
            "log": str(log_path.relative_to(ROOT)),
            "metrics": str((metrics / tag).relative_to(ROOT)),
            "experiment_fingerprint": fingerprint,
        }
        atomic_json(record_path, payload)
        if timed_out:
            status = "timeout"
        elif returncode == 0 and summary.get("status") == "completed":
            status = "completed"
        elif summary.get("status"):
            status = summary["status"]
        else:
            status = f"rc{returncode}"
        return tag, status

    completed = 0
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(run, job) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                tag, status = future.result()
                completed += 1
                print(f"[{completed:3d}/{len(jobs):3d}] {status:10s} {tag}", flush=True)
    finally:
        running_lock.unlink(missing_ok=True)

    print(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
