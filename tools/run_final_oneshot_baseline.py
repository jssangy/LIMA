#!/usr/bin/env python3
"""Run final disappear-at-target CBS or PRIMAL2 one-shot baselines."""

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
FREEZE_MANIFEST = ROOT / "results/reference_instantiation_freeze_v1/FINAL_MANIFEST.json"
DENSITIES = (1, 5, 10, 20, 30, 40, 50, 60, 65, 70, 75)


@dataclass(frozen=True)
class Instance:
    map_file: str
    scenario_template: str
    tiles: int


INSTANCES = {
    "warehouse_10_20": Instance(
        "data/maps/warehouse_10_20_paper.map",
        "data/scenarios/warehouse-10-20-paper/warehouse-10-20-paper_s{s}.scen", 2649),
    "warehouse_20_40": Instance(
        "data/maps/warehouse_20_40_paper.map",
        "data/scenarios/warehouse-20-40-paper/warehouse-20-40-paper_s{s}.scen", 10499),
    "cross_3030": Instance(
        "data/maps/cross_3030_paper.map",
        "data/scenarios/cross-30-30-paper/cross-30-30-paper_s{s}.scen", 10200),
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
    parser.add_argument("--algorithm", required=True, choices=("cbs", "primal2"))
    parser.add_argument("--maps", default=",".join(INSTANCES))
    parser.add_argument("--densities", default=",".join(map(str, DENSITIES)))
    parser.add_argument("--scenarios", default="0-9")
    parser.add_argument("--wall-budget", type=float, default=300.0)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--output-dir")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--freeze-manifest", default=str(FREEZE_MANIFEST))
    parser.add_argument("--cbs")
    parser.add_argument(
        "--primal-python", default=str(Path.home() / "miniconda3/envs/primal2/bin/python"))
    parser.add_argument(
        "--primal-script", default=str(Path.home() / "mapf-baselines/PRIMAL2/run_our_instances.py"))
    args = parser.parse_args()

    if args.jobs < 1 or args.wall_budget <= 0 or args.max_steps < 1:
        parser.error("jobs, wall-budget, and max-steps must be positive")
    maps = [item.strip() for item in args.maps.split(",") if item.strip()]
    if not maps or not set(maps).issubset(INSTANCES):
        parser.error("unknown or empty map selection")
    densities = parse_int_list(args.densities, set(DENSITIES))
    scenarios = parse_int_list(args.scenarios, set(range(10)))

    freeze_manifest = Path(args.freeze_manifest).resolve()
    freeze = json.loads(freeze_manifest.read_text(encoding="utf-8"))
    if freeze.get("status") != "frozen":
        parser.error("freeze manifest is not frozen")
    cbs_artifact = freeze["artifacts"]["cbs_binary"]
    cbs = (ROOT / (args.cbs or cbs_artifact["path"])).resolve()
    primal_python = Path(args.primal_python).resolve()
    primal_script = Path(args.primal_script).resolve()
    executable_files = [cbs] if args.algorithm == "cbs" else [primal_python, primal_script]
    for path in executable_files:
        if not path.is_file():
            parser.error(f"missing executable or adapter: {path}")
    if args.algorithm == "cbs" and args.cbs is None and sha256(cbs) != cbs_artifact["sha256"]:
        parser.error("CBS binary SHA-256 does not match the freeze manifest")

    output = (ROOT / (args.output_dir or
        f"results/revision_final/oneshot_{args.algorithm}_standard_v1")).resolve()
    records = output / "records"
    resources = output / "resources"
    logs = output / "logs"
    for directory in (records, resources, logs):
        directory.mkdir(parents=True, exist_ok=True)

    jobs: list[dict] = []
    inputs: dict[str, dict[str, object]] = {}
    for map_name in maps:
        spec = INSTANCES[map_name]
        map_path = ROOT / spec.map_file
        inputs[spec.map_file] = {"sha256": sha256(map_path), "size_bytes": map_path.stat().st_size}
        for scenario in scenarios:
            scenario_file = spec.scenario_template.format(s=scenario)
            scenario_path = ROOT / scenario_file
            inputs[scenario_file] = {
                "sha256": sha256(scenario_path), "size_bytes": scenario_path.stat().st_size}
        for density in densities:
            agents = density * spec.tiles // 100
            for scenario in scenarios:
                jobs.append({
                    "map": map_name, "density": density, "agents": agents,
                    "scenario": scenario, "map_file": spec.map_file,
                    "scenario_file": spec.scenario_template.format(s=scenario),
                    "tag": f"{map_name}_d{density:02d}_a{agents}_s{scenario}",
                })

    runner = Path(__file__).resolve()
    executable_hashes = {str(path): sha256(path) for path in executable_files}
    fingerprint_payload = {
        "schema_version": 1,
        "semantic_scope": "one-shot; disappear at physical sink",
        "algorithm": args.algorithm,
        "executables": executable_hashes,
        "runner_sha256": sha256(runner),
        "maps": maps, "densities": densities, "scenarios": scenarios,
        "wall_budget_seconds": args.wall_budget, "max_steps": args.max_steps,
        "inputs": {path: item["sha256"] for path, item in sorted(inputs.items())},
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
        "freeze_manifest": str(freeze_manifest.relative_to(ROOT)),
        "job_count": len(jobs), "jobs_concurrency": args.jobs,
        "host": {"hostname": socket.gethostname(), "platform": platform.platform(),
                 "cpu_count": os.cpu_count()},
        "input_files": inputs,
    })

    lock = output / ".RUNNING"
    if lock.exists():
        parser.error(f"campaign lock already exists: {lock}")
    lock.write_text(f"pid={os.getpid()}\nstarted_utc={datetime.now(timezone.utc).isoformat()}\n")

    def run(job: dict) -> tuple[str, str]:
        tag = job["tag"]
        record_path = records / f"{tag}_{args.algorithm}.json"
        if record_path.exists() and not args.rerun:
            return tag, "skipped"
        resource_path = resources / f"{tag}_{args.algorithm}.txt"
        log_path = logs / f"{tag}_{args.algorithm}.log"
        resource_path.unlink(missing_ok=True)
        if args.algorithm == "cbs":
            solver = [
                str(cbs), "--map", job["map_file"], "--scenario", job["scenario_file"],
                "--agents", str(job["agents"]), "--time-limit", str(args.wall_budget),
            ]
        else:
            solver = [
                str(primal_python), str(primal_script),
                "--map", str((ROOT / job["map_file"]).resolve()),
                "--scen", str((ROOT / job["scenario_file"]).resolve()),
                "-n", str(job["agents"]), "--seed", str(1234 + job["scenario"]),
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
        log_path.write_text(stdout + ("\n[stderr]\n" + stderr if stderr else ""), encoding="utf-8")
        payload = {
            **job, "algorithm": args.algorithm,
            "semantic_scope": "one-shot; disappear at physical sink",
            "returncode": returncode, "timed_out": timed_out,
            "runner_wall_seconds": time.time() - started,
            "result": result, "resource": parse_resource(resource_path),
            "command": command, "log": str(log_path.relative_to(ROOT)),
            "experiment_fingerprint": fingerprint,
        }
        atomic_json(record_path, payload)
        if timed_out:
            status = "timeout"
        else:
            status = "solved" if result.get("solved") == "1" else "unsolved"
        return tag, status

    completed = 0
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(run, job) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                tag, status = future.result()
                completed += 1
                print(f"[{completed:3d}/{len(jobs):3d}] {status:8s} {tag}", flush=True)
    finally:
        lock.unlink(missing_ok=True)
    print(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
