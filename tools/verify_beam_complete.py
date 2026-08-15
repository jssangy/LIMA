#!/usr/bin/env python3
"""Verify that beam-complete preserves successful frozen-beam results."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SPECS = (
    ("2_10_2_10_n14", "2,10,2,10", 14),
    ("10_2_10_n12", "10,2,10", 12),
    ("5_5_5_5_n15", "5,5,5,5", 15),
)
SOLVERS = ("beam", "beam-complete")
INSTANCES = 500
IGNORED_IDENTITY_FIELDS = {"wall_us"}


def rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def run_one(binary: Path, output: Path, label: str, arms: str, n: int, solver: str) -> Path:
    target = output / f"{'complete' if solver == 'beam-complete' else 'beam'}_{label}.csv"
    if len(rows(target)) == INSTANCES:
        return target
    command = [
        str(binary), "--mode", "bench", "--bench-arms", arms, "--bench-n", str(n),
        "--bench-instances", str(INSTANCES), "--seed", "7", "--solver", solver,
        "--beam-width", "2048", "--beam-score", "tt",
        "--solver-iterations", "2000000", "--output", str(target),
    ]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    (output / f"{'complete' if solver == 'beam-complete' else 'beam'}_{label}.log").write_text(
        completed.stdout + completed.stderr, encoding="utf-8"
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{solver} {label} exited {completed.returncode}")
    if len(rows(target)) != INSTANCES:
        raise RuntimeError(f"{target} does not contain {INSTANCES} instances")
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="build_gated_broad/lima")
    parser.add_argument("--output", default="results/phase2_gateb_beam_complete_identity")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    binary = (ROOT / args.binary).resolve()
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not binary.is_file():
        parser.error(f"binary does not exist: {binary}")

    jobs = [
        (binary, output, label, arms, n, solver)
        for label, arms, n in SPECS for solver in SOLVERS
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(lambda values: run_one(*values), jobs))

    comparisons = []
    mismatches = []
    fallback_total = 0
    for label, arms, n in SPECS:
        beam_rows = rows(output / f"beam_{label}.csv")
        complete_rows = rows(output / f"complete_{label}.csv")
        shape_mismatches = 0
        shape_fallbacks = sum(int(row["fallback"]) for row in complete_rows)
        fallback_total += shape_fallbacks
        for index, (beam_row, complete_row) in enumerate(zip(beam_rows, complete_rows)):
            fields = set(beam_row) | set(complete_row)
            differences = {
                field: (beam_row.get(field), complete_row.get(field))
                for field in fields - IGNORED_IDENTITY_FIELDS
                if beam_row.get(field) != complete_row.get(field)
            }
            if differences:
                shape_mismatches += 1
                if len(mismatches) < 20:
                    mismatches.append({"shape": label, "instance": index, "differences": differences})
        comparisons.append({
            "shape": label,
            "arms": arms,
            "n": n,
            "instances": len(beam_rows),
            "mismatches_excluding_wall_us": shape_mismatches,
            "complete_fallbacks": shape_fallbacks,
        })

    report = {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "binary": str(binary),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "seed": 7,
        "ignored_identity_fields": sorted(IGNORED_IDENTITY_FIELDS),
        "comparisons": comparisons,
        "total_instances": sum(row["instances"] for row in comparisons),
        "total_mismatches_excluding_wall_us": sum(row["mismatches_excluding_wall_us"] for row in comparisons),
        "total_complete_fallbacks": fallback_total,
        "mismatch_examples": mismatches,
    }
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return int(bool(report["total_mismatches_excluding_wall_us"] or fallback_total))


if __name__ == "__main__":
    raise SystemExit(main())
