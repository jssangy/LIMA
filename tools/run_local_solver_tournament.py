#!/usr/bin/env python3
"""Gate A: deterministic single-intersection solver tournament.

The simulator maps contain only three distinct local arm-capacity shapes.
For each shape this runner sweeps every feasible population from one agent to
the LIMA isolation bound (sum(capacities) - max(capacities)).  All solvers see
the same random instances and run one process at a time by default.  Aggregate
user CPU time, rather than contended wall time, is the speed metric.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import signal
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SHAPES = {
    "warehouse4": (2, 10, 2, 10),
    "warehouse3": (10, 2, 10),
    "cross4": (5, 5, 5, 5),
}
SOLVERS = {
    "ida_legacy": ("--solver-nodes", "2000000"),
    "ida_tt": ("--lb-mode", "tt", "--solver-nodes", "2000000"),
    "ida_ttdom": (
        "--lb-mode", "tt", "--dominance", "--solver-nodes", "2000000"
    ),
    "beam": ("--solver", "beam", "--solver-iterations", "2000000"),
    "greedy": ("--solver", "greedy"),
    "hybrid100": (
        "--solver", "hybrid", "--lb-mode", "tt", "--dominance",
        "--solver-nodes", "100", "--solver-iterations", "2000000",
    ),
    "hybrid1k": (
        "--solver", "hybrid", "--lb-mode", "tt", "--dominance",
        "--solver-nodes", "1000", "--solver-iterations", "2000000",
    ),
    "hybrid10k": (
        "--solver", "hybrid", "--lb-mode", "tt", "--dominance",
        "--solver-nodes", "10000", "--solver-iterations", "2000000",
    ),
    "hybrid100k": (
        "--solver", "hybrid", "--lb-mode", "tt", "--dominance",
        "--solver-nodes", "100000", "--solver-iterations", "2000000",
    ),
}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_text(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def parse_resource(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    return dict(re.findall(
        r"^(\w+)=([^\n]+)$", path.read_text(encoding="utf-8", errors="replace"), re.MULTILINE
    ))


def summarize_csv(path: Path) -> dict:
    if not path.is_file():
        return {"rows": 0, "outcomes": {}}
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    outcomes = Counter(row["outcome"] for row in rows)
    solved = [row for row in rows if row["outcome"] == "solved"]

    def percentile(values: list[int], p: float) -> int | None:
        if not values:
            return None
        values.sort()
        return values[min(len(values) - 1, int((len(values) - 1) * p))]

    nodes = [int(row["expanded"]) for row in rows]
    lengths = [int(row["solution_len"]) for row in solved]
    return {
        "rows": len(rows),
        "outcomes": dict(outcomes),
        "solved": len(solved),
        "solve_rate": len(solved) / len(rows) if rows else 0.0,
        "expanded_median": percentile(nodes, 0.50),
        "expanded_p90": percentile(nodes, 0.90),
        "expanded_p99": percentile(nodes, 0.99),
        "solution_len_median": percentile(lengths, 0.50),
        "solution_len_p90": percentile(lengths, 0.90),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="build_phase2/lima")
    parser.add_argument("--solvers", default=",".join(SOLVERS))
    parser.add_argument("--shapes", default=",".join(SHAPES))
    parser.add_argument("--instances", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=900.0,
                        help="per-cell safety watchdog; not an evaluation score")
    parser.add_argument("--output-dir", default="results/phase2_local_solver")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    solvers = [name for name in args.solvers.split(",") if name]
    shapes = [name for name in args.shapes.split(",") if name]
    if not solvers or not set(solvers).issubset(SOLVERS):
        parser.error(f"solvers must be a subset of {sorted(SOLVERS)}")
    if not shapes or not set(shapes).issubset(SHAPES):
        parser.error(f"shapes must be a subset of {sorted(SHAPES)}")
    if args.instances < 1:
        parser.error("instances must be positive")

    binary = (ROOT / args.binary).resolve()
    if not binary.is_file():
        parser.error(f"binary does not exist: {binary}")
    output = ROOT / args.output_dir
    records = output / "records"
    raw = output / "raw"
    resources = output / "resources"
    records.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    resources.mkdir(parents=True, exist_ok=True)

    jobs = []
    for shape in shapes:
        capacities = SHAPES[shape]
        bound = sum(capacities) - max(capacities)
        for population in range(1, bound + 1):
            for solver in solvers:
                jobs.append((shape, capacities, bound, population, solver))

    atomic_json(output / "MANIFEST.json", {
        "purpose": "Phase 2 Gate A single-intersection solver freeze",
        "evaluation": "solve rate, expanded nodes, solution length, aggregate user CPU",
        "concurrency": 1,
        "binary": str(binary.relative_to(ROOT)),
        "binary_sha256": sha256(binary),
        "git_head": git_text("rev-parse", "HEAD"),
        "git_status": git_text("status", "--short"),
        "solvers": solvers,
        "solver_flags": {name: list(SOLVERS[name]) for name in solvers},
        "shapes": {name: list(SHAPES[name]) for name in shapes},
        "population_rule": "every N from 1 through sum(arms)-max(arms)",
        "instances_per_cell": args.instances,
        "seed": args.seed,
        "watchdog_seconds": args.timeout,
        "job_count": len(jobs),
    })

    for index, (shape, capacities, bound, population, solver) in enumerate(jobs, 1):
        tag = f"{shape}_n{population:02d}_{solver}"
        record_path = records / f"{tag}.json"
        if record_path.is_file() and not args.rerun:
            print(f"[{index:3d}/{len(jobs):3d}] skipped  {tag}", flush=True)
            continue
        csv_path = raw / f"{tag}.csv"
        resource_path = resources / f"{tag}.txt"
        resource_path.unlink(missing_ok=True)
        command = [
            "/usr/bin/time", "-f", "user_seconds=%U\nsystem_seconds=%S\nmax_rss_kb=%M",
            "-o", str(resource_path), str(binary), "--mode", "bench",
            "--bench-arms", ",".join(str(value) for value in capacities),
            "--bench-n", str(population), "--bench-instances", str(args.instances),
            "--seed", str(args.seed), *SOLVERS[solver], "--output", str(csv_path),
        ]
        started = time.time()
        timed_out = False
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=args.timeout)
            returncode = process.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
            returncode = 124
        resource = parse_resource(resource_path)
        summary = summarize_csv(csv_path)
        try:
            user_seconds = float(resource.get("user_seconds", "nan"))
        except ValueError:
            user_seconds = float("nan")
        if summary["rows"]:
            summary["user_cpu_us_per_instance"] = user_seconds * 1e6 / summary["rows"]
        atomic_json(record_path, {
            "tag": tag,
            "shape": shape,
            "capacities": capacities,
            "bound": bound,
            "population": population,
            "bound_density": population / bound,
            "solver": solver,
            "instances_requested": args.instances,
            "returncode": returncode,
            "timed_out": timed_out,
            "runner_wall_seconds": time.time() - started,
            "resource": resource,
            "summary": summary,
            "command": command,
            "stdout_tail": stdout[-1000:],
            "stderr_tail": stderr[-1000:],
        })
        status = "watchdog" if timed_out else f"{summary.get('solved', 0)}/{summary.get('rows', 0)}"
        print(f"[{index:3d}/{len(jobs):3d}] {status:9s} {tag}", flush=True)

    print(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
