#!/usr/bin/env python3
"""Verify the frozen LIMA reference-instantiation contract.

This is a fast preflight, not a performance experiment.  It checks that the
binary was built from the clean current commit, that ``lima-default`` exposes
the versioned component bundle in its provenance, that explicit component
overrides are independent of CLI order, and that two identical executions are
deterministic and conflict-free.  A machine-readable report is written for
the experiment handoff.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config/reference_instantiation_v2.json"
DETERMINISTIC_FIELDS = (
    "status",
    "steps",
    "completed",
    "moves",
    "waits",
    "deadlocks",
    "intersections",
    "validation",
    "vertex_conflicts",
    "edge_conflicts",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fields(text: str) -> dict[str, str]:
    lines = [line for line in text.splitlines() if line.strip()]
    return dict(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)", lines[-1])) if lines else {}


def run(command: list[str], *, expected_codes: tuple[int, ...] = (0, 2)) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=300)
    if completed.returncode not in expected_codes:
        raise RuntimeError(
            f"command exited {completed.returncode}: {' '.join(command)}\n"
            f"stdout: {completed.stdout[-1000:]}\nstderr: {completed.stderr[-1000:]}"
        )
    return completed


def solve_command(
    binary: Path, solution: Path, jsonl_trace: Path, extra: list[str], profile_last: bool = False
) -> list[str]:
    base = [
        str(binary),
        "--mode", "solve",
        "--map", "data/maps/warehouse_10_20_paper.map",
        "--scenario", "data/scenarios/warehouse-10-20-paper/warehouse-10-20-paper_s0.scen",
        "--agents", "26",
        "--planner", "bfs",
        "--seed", "0",
        "--max-steps", "2000",
        "--output", str(solution),
        "--trace-jsonl", str(jsonl_trace),
        "--validate-conflicts",
    ]
    if profile_last:
        return [*base, *extra, "--profile", "lima-default"]
    return [*base, "--profile", "lima-default", *extra]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="build_frozen/lima")
    parser.add_argument("--output", default="results/reference_instantiation_freeze_v1")
    parser.add_argument("--allow-dirty", action="store_true", help="allow a dirty pre-commit build")
    args = parser.parse_args()

    binary = (ROOT / args.binary).resolve()
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not binary.is_file():
        parser.error(f"binary does not exist: {binary}")

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    expected = config["expected_provenance"]
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    git_short = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    git_status = subprocess.check_output(
        ["git", "status", "--short", "--untracked-files=no"], cwd=ROOT, text=True
    ).strip()
    version_result = run([str(binary), "--version"], expected_codes=(0,))
    version = fields(version_result.stdout)
    expected_commit = f"{git_short}-dirty" if git_status else git_short
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check("tracked_worktree_clean", not git_status or args.allow_dirty, git_status or "clean")
    check("binary_commit_matches_source", version.get("commit") == expected_commit, {
        "expected": expected_commit,
        "actual": version.get("commit"),
    })
    check("binary_profile_version", version.get("profile_version") == str(config["profile_version"]), version)

    summaries: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        temp = Path(temporary)
        for index in range(2):
            solution = temp / f"repeat_{index}.txt"
            trace = temp / f"repeat_{index}.jsonl"
            completed = run(solve_command(binary, solution, trace, []))
            summary = fields(completed.stdout)
            summaries.append(summary)
            trace_check = run(
                ["python3", "tools/verify_trace.py", str(trace)], expected_codes=(0,)
            )
            check(f"trace_{index}_valid", True, trace_check.stdout.strip())

        first = summaries[0]
        provenance_mismatches = {
            key: {"expected": value, "actual": first.get(key)}
            for key, value in expected.items()
            if first.get(key) != value
        }
        check("profile_contract", not provenance_mismatches, provenance_mismatches or "exact match")
        safety = {
            "validation": "ok",
            "vertex_conflicts": "0",
            "edge_conflicts": "0",
        }
        safety_mismatches = {
            key: {"expected": value, "actual": first.get(key)}
            for key, value in safety.items()
            if first.get(key) != value
        }
        check("vertex_edge_safety", not safety_mismatches, safety_mismatches or "ok")
        deterministic_mismatches = {
            key: [summaries[0].get(key), summaries[1].get(key)]
            for key in DETERMINISTIC_FIELDS
            if summaries[0].get(key) != summaries[1].get(key)
        }
        check("repeat_determinism", not deterministic_mismatches,
              deterministic_mismatches or "all deterministic summary fields match")

        overrides = ["--solver", "beam", "--routing", "direct", "--capacity-formula", "plus-one"]
        order_summaries = []
        for profile_last in (False, True):
            solution = temp / f"order_{int(profile_last)}.txt"
            trace = temp / f"order_{int(profile_last)}.jsonl"
            order_summaries.append(fields(run(
                solve_command(binary, solution, trace, overrides, profile_last=profile_last)
            ).stdout))
        order_expected = {"solver": "beam", "routing": "direct", "capacity": "plus-one"}
        order_mismatches = {
            key: [order_summaries[0].get(key), order_summaries[1].get(key), value]
            for key, value in order_expected.items()
            if order_summaries[0].get(key) != value or order_summaries[1].get(key) != value
        }
        deterministic_order_mismatches = {
            key: [order_summaries[0].get(key), order_summaries[1].get(key)]
            for key in DETERMINISTIC_FIELDS
            if order_summaries[0].get(key) != order_summaries[1].get(key)
        }
        check("profile_override_order_independent",
              not order_mismatches and not deterministic_order_mismatches,
              {"provenance": order_mismatches, "trajectory": deterministic_order_mismatches}
              if order_mismatches or deterministic_order_mismatches else "exact match")

    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(CONFIG.relative_to(ROOT)),
        "config_sha256": sha256(CONFIG),
        "git_head": git_head,
        "git_status_tracked": git_status,
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "version": version,
        "checks": checks,
        "passed": all(item["passed"] for item in checks),
    }
    report_path = output / "verification.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"report={report_path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
