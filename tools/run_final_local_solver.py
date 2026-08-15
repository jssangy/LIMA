#!/usr/bin/env python3
"""Measure the frozen reference local solver on the operational envelope."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import socket
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
FREEZE_MANIFEST = ROOT / "results/reference_instantiation_freeze_v1/FINAL_MANIFEST.json"
SHAPES = {
    "warehouse_10_20": (2, 10, 2, 10),
    "warehouse_20_40": (10, 2, 10),
    "cross_3030": (5, 5, 5, 5),
}
LOADS = {"p50": 0.50, "p75": 0.75, "p100": 1.00}


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


def parse_resource(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    return dict(re.findall(
        r"^(\w+)=([^\n]+)$", path.read_text(encoding="utf-8", errors="replace"), re.MULTILINE))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def summarize(path: Path, requested: int) -> dict:
    if not path.is_file():
        return {"rows": 0, "requested": requested}
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    solved = [row for row in rows if row["outcome"] == "solved"]
    def distribution(field: str) -> dict[str, float]:
        values = [float(row[field]) for row in rows]
        return {
            "median": percentile(values, 0.50),
            "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99),
            "maximum": max(values) if values else math.nan,
        }
    return {
        "rows": len(rows),
        "requested": requested,
        "solved": len(solved),
        "solved_rate": len(solved) / len(rows) if rows else 0.0,
        "outcomes": {name: sum(row["outcome"] == name for row in rows)
                     for name in sorted({row["outcome"] for row in rows})},
        "fallbacks": sum(int(row["fallback"]) for row in rows),
        "fallback_rate": sum(int(row["fallback"]) for row in rows) / len(rows) if rows else 0.0,
        "fastpaths": sum(int(row["fastpath"]) for row in rows),
        "wall_us": distribution("wall_us"),
        "expanded": distribution("expanded"),
        "iterations": distribution("iterations"),
        "solution_len": distribution("solution_len"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-manifest", default=str(FREEZE_MANIFEST))
    parser.add_argument("--instances", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--output-dir", default="results/revision_final/local_solver_reference_step_v2")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    if args.instances < 1:
        parser.error("instances must be positive")

    freeze_path = Path(args.freeze_manifest).resolve()
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    artifact = freeze["artifacts"]["lima_binary"]
    binary = (ROOT / artifact["path"]).resolve()
    if freeze.get("status") != "frozen" or not binary.is_file() or sha256(binary) != artifact["sha256"]:
        parser.error("frozen LIMA artifact missing, mismatched, or not frozen")

    output = (ROOT / args.output_dir).resolve()
    raw, records, resources, logs = (
        output / "raw", output / "records", output / "resources", output / "logs")
    for directory in (raw, records, resources, logs):
        directory.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).resolve()
    jobs = []
    for shape, capacities in SHAPES.items():
        bound = sum(capacities) - max(capacities)
        for load, fraction in LOADS.items():
            population = math.floor(fraction * bound)
            jobs.append({"shape": shape, "capacities": capacities, "bound": bound,
                         "load": load, "fraction": fraction, "population": population,
                         "tag": f"{shape}_{load}_n{population}"})
    fingerprint_payload = {
        "schema_version": 2,
        "semantic_scope": "synthetic single-intersection distribution; operational capacity",
        "freeze_commit": freeze["git_commit"],
        "binary_sha256": sha256(binary),
        "runner_sha256": sha256(runner),
        "instances_per_cell": args.instances,
        "seed": args.seed,
        "termination_policy": "fixed instance count; no wall-clock cutoff",
        "jobs": jobs,
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
        "host": {"hostname": socket.gethostname(), "platform": platform.platform(),
                 "cpu_count": os.cpu_count()},
    })

    lock = output / ".RUNNING"
    if lock.exists():
        parser.error(f"campaign lock already exists: {lock}")
    lock.write_text(f"pid={os.getpid()}\nstarted_utc={datetime.now(timezone.utc).isoformat()}\n")
    try:
        for index, job in enumerate(jobs, 1):
            tag = job["tag"]
            record_path = records / f"{tag}.json"
            if record_path.exists() and not args.rerun:
                print(f"[{index}/9] skipped {tag}", flush=True)
                continue
            csv_path = raw / f"{tag}.csv"
            resource_path = resources / f"{tag}.txt"
            log_path = logs / f"{tag}.log"
            resource_path.unlink(missing_ok=True)
            command = [
                "/usr/bin/time", "-f",
                "max_rss_kb=%M\nuser_seconds=%U\nsystem_seconds=%S\nelapsed_seconds=%e",
                "-o", str(resource_path), str(binary), "--profile", "lima-default",
                "--mode", "bench", "--bench-arms", ",".join(map(str, job["capacities"])),
                "--bench-n", str(job["population"]), "--bench-instances", str(args.instances),
                "--seed", str(args.seed), "--output", str(csv_path),
            ]
            started = time.time()
            proc = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, start_new_session=True)
            stdout, stderr = proc.communicate()
            returncode = proc.returncode
            resource = parse_resource(resource_path)
            summary = summarize(csv_path, args.instances)
            try:
                user_seconds = float(resource.get("user_seconds", "nan"))
            except ValueError:
                user_seconds = math.nan
            summary["user_cpu_us_per_instance"] = (
                user_seconds * 1e6 / summary["rows"] if summary["rows"] else math.nan)
            log_path.write_text(stdout + ("\n[stderr]\n" + stderr if stderr else ""), encoding="utf-8")
            atomic_json(record_path, {
                **job, "returncode": returncode, "timed_out": False,
                "runner_wall_seconds": time.time() - started, "summary": summary,
                "resource": resource, "command": command, "raw_csv": str(csv_path.relative_to(ROOT)),
                "log": str(log_path.relative_to(ROOT)), "experiment_fingerprint": fingerprint,
            })
            status = f"{summary['solved']}/{summary['rows']}"
            print(f"[{index}/9] {status:9s} {tag}", flush=True)
    finally:
        lock.unlink(missing_ok=True)

    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(records.glob("*.json"))]
    fields = ["shape", "load", "population", "bound", "rows", "solved", "solved_rate",
              "fallback_rate", "solution_len_median", "solution_len_p95", "wall_us_median",
              "wall_us_p95", "wall_us_p99", "wall_us_maximum", "expanded_median",
              "expanded_p95", "user_cpu_us_per_instance", "max_rss_kb", "timed_out"]
    summary_csv = output / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in rows:
            stats = record["summary"]
            writer.writerow({
                "shape": record["shape"], "load": record["load"],
                "population": record["population"], "bound": record["bound"],
                "rows": stats["rows"], "solved": stats.get("solved", 0),
                "solved_rate": stats.get("solved_rate", 0),
                "fallback_rate": stats.get("fallback_rate", 0),
                "solution_len_median": stats.get("solution_len", {}).get("median"),
                "solution_len_p95": stats.get("solution_len", {}).get("p95"),
                "wall_us_median": stats.get("wall_us", {}).get("median"),
                "wall_us_p95": stats.get("wall_us", {}).get("p95"),
                "wall_us_p99": stats.get("wall_us", {}).get("p99"),
                "wall_us_maximum": stats.get("wall_us", {}).get("maximum"),
                "expanded_median": stats.get("expanded", {}).get("median"),
                "expanded_p95": stats.get("expanded", {}).get("p95"),
                "user_cpu_us_per_instance": stats.get("user_cpu_us_per_instance"),
                "max_rss_kb": record["resource"].get("max_rss_kb"),
                "timed_out": record["timed_out"],
            })
    markdown = [
        "# Frozen reference single-intersection distribution", "",
        f"- Cells: {len(rows)}/9", f"- Instances: {sum(r['summary']['rows'] for r in rows)}/900",
        f"- Unsolved: {sum(r['summary']['rows'] - r['summary'].get('solved', 0) for r in rows)}",
        f"- Fallbacks: {sum(r['summary'].get('fallbacks', 0) for r in rows)}", "",
        "| shape | load | n/B | solved | CPU median/p95/p99/max (us) | schedule median/p95 | fallback | RSS MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in rows:
        stats = record["summary"]
        wall = stats.get("wall_us", {})
        solution = stats.get("solution_len", {})
        rss = float(record["resource"].get("max_rss_kb", "nan")) / 1024
        markdown.append(
            f"| {record['shape']} | {record['load']} | {record['population']}/{record['bound']} | "
            f"{stats.get('solved', 0)}/{stats['rows']} | "
            f"{wall.get('median', math.nan):.0f}/{wall.get('p95', math.nan):.0f}/"
            f"{wall.get('p99', math.nan):.0f}/{wall.get('maximum', math.nan):.0f} | "
            f"{solution.get('median', math.nan):.1f}/{solution.get('p95', math.nan):.1f} | "
            f"{stats.get('fallback_rate', 0):.3f} | {rss:.1f} |")
    (output / "SUMMARY.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(summary_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
