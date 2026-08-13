#!/usr/bin/env python3
"""Golden regression: run the quick cell set and compare deterministic fields.

Usage: tools/regress.py [--binary build/lima] [--jobs 2] [--golden tests/golden/e0_quick.golden]

Every golden cell is re-run with the same map/agents/seed and the deterministic
summary fields must match exactly.  Any mismatch means the change is not
behavior-preserving and must be either fixed or explicitly re-baselined.

Golden line formats (both accepted):
  map|agents|seed|fields            legacy random mode (--agents/--seed)
  map|scen_dir|agents|seed|fields   scenario mode when scen_dir is non-empty:
                                    tasks come from data/scenarios/<scen_dir>/
                                    <scen_dir>_s<seed>.scen (seed doubles as
                                    the scenario rollout index and is still
                                    passed as --seed); an empty scen_dir means
                                    random mode.
"""
import argparse
import concurrent.futures
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DETERMINISTIC = ("status", "steps", "completed", "moves", "waits", "deadlocks",
                 "intersections", "validation", "vertex_conflicts", "edge_conflicts")

def run_cell(binary: Path, map_name: str, scen_dir: str, agents: int, seed: int) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=True) as tmp:
        cmd = [str(binary), "--mode", "solve", "--map", f"data/maps/{map_name}",
               "--agents", str(agents), "--planner", "bfs", "--seed", str(seed),
               "--output", tmp.name, "--validate-conflicts"]
        if scen_dir:
            cmd += ["--scenario", f"data/scenarios/{scen_dir}/{scen_dir}_s{seed}.scen"]
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=3600)
    # exit 0 = all tasks completed, exit 2 = run ended without completing
    # (step limit / all-stalled); both print a deterministic summary line and
    # are valid golden outcomes -- the status field records which one.
    if proc.returncode not in (0, 2):
        raise RuntimeError(f"exit code {proc.returncode}; stderr tail: {proc.stderr.strip()[-300:]}")
    summary = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    return dict(re.findall(r"(\w+)=([^ ]+)", summary))

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="build/lima")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--golden", default="tests/golden/e0_quick.golden")
    args = parser.parse_args()

    binary = (ROOT / args.binary).resolve()
    golden_path = ROOT / args.golden
    cells = []
    for line in golden_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) == 4:          # legacy: map|agents|seed|fields
            map_name, agents, seed, kept = parts
            scen_dir = ""
        elif len(parts) == 5:        # map|scen_dir|agents|seed|fields
            map_name, scen_dir, agents, seed, kept = parts
        else:
            raise ValueError(f"malformed golden line: {line}")
        cells.append((map_name, scen_dir, int(agents), int(seed),
                      dict(re.findall(r"(\w+)=([^ ]+)", kept))))

    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_cell, binary, m, sc, a, s): (m, sc, a, s, want)
                   for (m, sc, a, s, want) in cells}
        for future in concurrent.futures.as_completed(futures):
            m, sc, a, s, want = futures[future]
            label = f"{m} {sc or 'random'} a{a} s{s}"
            try:
                got = future.result()
            except Exception as error:  # noqa: BLE001 - report and continue
                failures.append((label, f"run error: {error}"))
                continue
            diffs = [f"{k}: want {want[k]} got {got.get(k, '<missing>')}"
                     for k in DETERMINISTIC if k in want and got.get(k) != want[k]]
            if diffs:
                failures.append((label, "; ".join(diffs)))
            print(f"{'FAIL' if diffs else 'ok  '} {label}")

    print(f"\n{len(cells) - len(failures)}/{len(cells)} cells identical")
    for label, why in failures:
        print(f"  FAIL {label}: {why}")
    return 1 if failures else 0

if __name__ == "__main__":
    sys.exit(main())
