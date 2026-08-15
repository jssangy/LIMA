#!/usr/bin/env python3
"""Run the resumable Gate C variant grid with bounded process concurrency.

Each variant is delegated to ``run_revision_grid.py`` and receives its own
manifest/record directory.  The outer and inner worker counts are recorded by
the child manifests only as execution metadata: Gate C ranks deterministic
simulation-step outcomes, never contended wall-clock time.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VARIANTS = (
    "gatec_base",
    "gatec_paper",
    "gatec_cap4",
    "gatec_cap8",
    "gatec_cap12",
    "gatec_margin1",
    "gatec_margin2",
    "gatec_hyst1",
    "gatec_hyst2",
    "gatec_subset",
    "gatec_noresync",
    "gatec_paper_margin1",
    "gatec_paper_hyst1",
    "gatec_paper_subset",
    "gatec_cap8_subset",
    "gatec_hyst1_subset",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="build_gatea_frozen/lima")
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--maps", default="warehouse_10_20,warehouse_20_40,cross_3030")
    parser.add_argument("--densities", default="1,5,10,20,30,40,50,60")
    parser.add_argument("--scenarios", default="0-1")
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--cell-timeout", type=float, default=86_400.0)
    parser.add_argument("--outer-jobs", type=int, default=6)
    parser.add_argument("--inner-jobs", type=int, default=2)
    parser.add_argument("--output-root", default="results/phase2_gatec_h10000")
    args = parser.parse_args()

    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    if not variants:
        parser.error("--variants must not be empty")
    if args.outer_jobs < 1 or args.inner_jobs < 1:
        parser.error("worker counts must be positive")
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")

    script = ROOT / "tools" / "run_revision_grid.py"
    output_root = ROOT / args.output_root

    def run(variant: str) -> tuple[str, int]:
        variant_output = output_root / variant
        variant_output.mkdir(parents=True, exist_ok=True)
        lock = variant_output / ".RUNNING"
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            print(f"variant={variant} already owned by another runner", flush=True)
            return variant, 0
        os.close(descriptor)
        command = [
            sys.executable,
            str(script),
            "--binary", args.binary,
            "--variant", variant,
            "--maps", args.maps,
            "--densities", args.densities,
            "--scenarios", args.scenarios,
            "--jobs", str(args.inner_jobs),
            "--timeout", str(args.cell_timeout),
            "--max-steps", str(args.max_steps),
            "--output-dir", str(variant_output),
        ]
        environment = os.environ.copy()
        environment["LIMA_GATE_RUNNER_OWNS_LOCK"] = "1"
        try:
            process = subprocess.run(command, cwd=ROOT, text=True, env=environment)
            return variant, process.returncode
        finally:
            lock.unlink(missing_ok=True)

    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.outer_jobs) as pool:
        futures = {pool.submit(run, variant): variant for variant in variants}
        for future in concurrent.futures.as_completed(futures):
            variant, returncode = future.result()
            print(f"variant={variant} returncode={returncode}", flush=True)
            if returncode != 0:
                failures.append(variant)

    if failures:
        print("failed variants: " + ", ".join(failures), file=sys.stderr)
        return 1
    print(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
