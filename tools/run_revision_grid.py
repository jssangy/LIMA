#!/usr/bin/env python3
"""Run resumable LIMA grids on the submitted-paper instance definition.

The canonical revision instance combines the unchanged warehouse geometries
with the submitted scenarios, and the submitted 187x187 Square-1 geometry with
its scenarios.  Agent counts follow Table 2 exactly: floor(density * Tiles).

Each cell is an atomic JSON record. Existing records are skipped, so an
interrupted grid can be resumed with the same command. Broad grids default to
``--no-trace``; use ``--record-trace`` only for selected validation cells.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DENSITIES = (1, 5, 10, 20, 30, 40, 50, 60)


@dataclass(frozen=True)
class Instance:
    map_file: str
    scenario_template: str
    tiles: int


INSTANCES = {
    "warehouse_10_20": Instance(
        "data/maps/warehouse_10_20.map",
        "data/scenarios/warehouse-10-20-paper/warehouse-10-20-paper_s{s}.scen",
        2649,
    ),
    "warehouse_20_40": Instance(
        "data/maps/warehouse_20_40.map",
        "data/scenarios/warehouse-20-40-paper/warehouse-20-40-paper_s{s}.scen",
        10499,
    ),
    "cross_3030": Instance(
        "data/maps/cross_3030_paper.map",
        "data/scenarios/cross-30-30-paper/cross-30-30-paper_s{s}.scen",
        10200,
    ),
}


VARIANTS = {
    "base": (),
    "tt": ("--lb-mode", "tt"),
    "ttdom": ("--lb-mode", "tt", "--dominance"),
    "nodes2m": ("--solver-nodes", "2000000"),
    "tt_nodes2m": ("--lb-mode", "tt", "--solver-nodes", "2000000"),
    "replan8": ("--pibt-replan", "8"),
    "replan8_tt_nodes2m": (
        "--pibt-replan", "8", "--lb-mode", "tt", "--solver-nodes", "2000000"
    ),
    "replan8_nodisc": ("--pibt-replan", "8", "--no-discharge"),
    "replan8_drandom": ("--pibt-replan", "8", "--discharge-random"),
    "lifelong": ("--goal-behavior", "lifelong"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_int_list(text: str, allowed: set[int] | None = None) -> list[int]:
    values: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(value) for value in part.split("-", 1))
            values.update(range(lo, hi + 1))
        else:
            values.add(int(part))
    if allowed is not None and not values.issubset(allowed):
        raise argparse.ArgumentTypeError(f"values must be a subset of {sorted(allowed)}")
    return sorted(values)


def git_text(*args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True
    )
    return proc.stdout.strip()


def parse_summary(text: str) -> dict[str, str]:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return {}
    return dict(re.findall(r"(\w+)=([^\s]+)", lines[-1]))


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="build_gating2/lima")
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="base")
    parser.add_argument("--maps", default=",".join(INSTANCES))
    parser.add_argument("--densities", default=",".join(str(v) for v in DENSITIES))
    parser.add_argument("--scenarios", default="0-1")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=2400.0)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--metrics", action="store_true")
    parser.add_argument("--record-trace", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    maps = [name.strip() for name in args.maps.split(",") if name.strip()]
    unknown = sorted(set(maps) - set(INSTANCES))
    if unknown:
        parser.error(f"unknown maps: {', '.join(unknown)}")
    densities = parse_int_list(args.densities, set(DENSITIES))
    scenarios = parse_int_list(args.scenarios, set(range(10)))
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.record_trace and args.timeout < 1:
        parser.error("--timeout must be positive")

    binary = (ROOT / args.binary).resolve()
    if not binary.is_file():
        parser.error(f"binary does not exist: {binary}")
    output = ROOT / (args.output_dir or f"results/revision_grid/{args.variant}")
    records = output / "records"
    metrics_root = output / "metrics"
    traces_root = output / "traces"
    records.mkdir(parents=True, exist_ok=True)

    instance_files: dict[str, dict[str, str]] = {}
    jobs = []
    for map_name in maps:
        instance = INSTANCES[map_name]
        map_path = ROOT / instance.map_file
        if not map_path.is_file():
            parser.error(f"missing map: {map_path}")
        instance_files[map_name] = {"map": instance.map_file, "map_sha256": sha256(map_path)}
        for density in densities:
            agents = density * instance.tiles // 100
            for scenario in scenarios:
                scenario_file = instance.scenario_template.format(s=scenario)
                scenario_path = ROOT / scenario_file
                if not scenario_path.is_file():
                    parser.error(f"missing scenario: {scenario_path}")
                tag = f"{map_name}_d{density:02d}_a{agents}_s{scenario}"
                jobs.append((map_name, density, agents, scenario, instance.map_file, scenario_file, tag))

    manifest = {
        "binary": str(binary.relative_to(ROOT)),
        "binary_sha256": sha256(binary),
        "git_head": git_text("rev-parse", "HEAD"),
        "git_status": git_text("status", "--short"),
        "variant": args.variant,
        "variant_flags": list(VARIANTS[args.variant]),
        "maps": maps,
        "densities": densities,
        "scenarios": scenarios,
        "timeout_seconds": args.timeout,
        "max_steps": args.max_steps,
        "metrics": args.metrics,
        "record_trace": args.record_trace,
        "instances": instance_files,
        "job_count": len(jobs),
    }
    write_json_atomic(output / "MANIFEST.json", manifest)

    def run(job) -> tuple[str, str]:
        map_name, density, agents, scenario, map_file, scenario_file, tag = job
        record = records / f"{tag}.json"
        if record.exists() and not args.rerun:
            return tag, "skipped"
        command = [
            str(binary), "--mode", "solve", "--map", map_file,
            "--scenario", scenario_file, "--agents", str(agents),
            "--planner", "bfs", "--seed", str(scenario),
            "--max-steps", str(args.max_steps), *VARIANTS[args.variant],
        ]
        if args.record_trace:
            traces_root.mkdir(parents=True, exist_ok=True)
            command += ["--output", str(traces_root / f"{tag}.txt"), "--validate-conflicts"]
        else:
            command.append("--no-trace")
        if args.metrics:
            command += ["--metrics", str(metrics_root / tag)]

        started = time.time()
        timed_out = False
        try:
            proc = subprocess.run(
                command, cwd=ROOT, capture_output=True, text=True, timeout=args.timeout
            )
            returncode = proc.returncode
            stdout = proc.stdout
            stderr = proc.stderr
        except subprocess.TimeoutExpired as error:
            timed_out = True
            returncode = 124
            stdout = error.stdout.decode() if isinstance(error.stdout, bytes) else (error.stdout or "")
            stderr = error.stderr.decode() if isinstance(error.stderr, bytes) else (error.stderr or "")
        payload = {
            "tag": tag,
            "map": map_name,
            "density_percent": density,
            "agents": agents,
            "scenario": scenario,
            "map_file": map_file,
            "scenario_file": scenario_file,
            "variant": args.variant,
            "command": command,
            "returncode": returncode,
            "timed_out": timed_out,
            "runner_wall_seconds": time.time() - started,
            "summary": parse_summary(stdout),
            "stdout_tail": stdout[-2000:],
            "stderr_tail": stderr[-2000:],
        }
        write_json_atomic(record, payload)
        status = "timeout" if timed_out else payload["summary"].get("status", f"rc{returncode}")
        return tag, status

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run, job): job[-1] for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            tag, status = future.result()
            completed += 1
            print(f"[{completed:3d}/{len(jobs):3d}] {status:10s} {tag}", flush=True)

    print(f"records: {records}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
