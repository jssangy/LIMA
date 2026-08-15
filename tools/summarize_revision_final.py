#!/usr/bin/env python3
"""Audit and summarize all final LIMA revision campaigns."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "results/revision_final"
CAMPAIGNS = {
    "oneshot_lima_standard_v1": 330,
    "oneshot_cbs_standard_v1": 330,
    "oneshot_primal2_standard_v1": 330,
    "oneshot_lima_certified_v1": 80,
    "oneshot_cbs_certified_v1": 80,
    "oneshot_primal2_certified_v1": 80,
    "stochastic_lima_v1": 360,
    "stochastic_primal2_v1": 360,
    "lifelong_lima_v1": 180,
    "admission_ablation_v1": 24,
    "local_solver_reference_v1": 9,
}


def numeric(value) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def result_fields(record: dict) -> dict:
    return record.get("summary") or record.get("result") or {}


def success(record: dict) -> bool:
    if record.get("timed_out"):
        return False
    if "horizon_completed" in record:
        return bool(record["horizon_completed"])
    if "solved" in record:
        return bool(record["solved"])
    fields = result_fields(record)
    if fields.get("status") == "completed" or fields.get("solved") in ("1", 1, True):
        return True
    if "rows" in fields and "solved" in fields:
        return int(fields["rows"]) == int(fields["solved"])
    return False


def makespan(record: dict) -> float | None:
    fields = result_fields(record)
    for key in ("makespan", "steps"):
        value = numeric(fields.get(key))
        if value is not None:
            return value
    return None


def group_label(record: dict) -> str:
    parts = [str(record.get("map") or record.get("shape") or "all")]
    for key in ("scope", "target", "density", "delay_probability", "probability", "variant", "load"):
        if key in record:
            parts.append(f"{key}={record[key]}")
    return "|".join(parts)


def main() -> int:
    audit = {}
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    all_complete = True
    for campaign, expected in CAMPAIGNS.items():
        root = OUTPUT / campaign
        records_dir = root / "records"
        records = []
        malformed = []
        for path in sorted(records_dir.glob("*.json")) if records_dir.is_dir() else []:
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError) as error:
                malformed.append({"file": str(path.relative_to(ROOT)), "error": str(error)})
        for record in records:
            grouped[(campaign, group_label(record))].append(record)
        timeouts = sum(bool(record.get("timed_out")) for record in records)
        nonzero = sum(record.get("returncode", 0) not in (0, None) for record in records)
        successes = sum(success(record) for record in records)
        fingerprints = sorted({record.get("experiment_fingerprint") for record in records
                               if record.get("experiment_fingerprint")})
        complete = len(records) == expected and not malformed and len(fingerprints) <= 1
        all_complete &= complete
        audit[campaign] = {
            "expected_records": expected, "records": len(records), "complete": complete,
            "successes": successes, "timeouts": timeouts, "nonzero_returncodes": nonzero,
            "malformed_records": malformed, "fingerprints": fingerprints,
            "manifest_present": (root / "MANIFEST.json").is_file(),
            "running_lock_present": (root / ".RUNNING").exists(),
            "runner_error_bytes": (root / "runner.err").stat().st_size
                if (root / "runner.err").is_file() else 0,
        }

    payload = {"all_complete": all_complete, "campaigns": audit}
    (OUTPUT / "FINAL_AUDIT.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                                             encoding="utf-8")
    lines = ["# LIMA revision final experiment audit", "",
             f"- Complete: {'yes' if all_complete else 'no'}", "",
             "| campaign | records | success | timeout | nonzero rc | malformed | lock |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for campaign, row in audit.items():
        lines.append(
            f"| {campaign} | {row['records']}/{row['expected_records']} | {row['successes']} | "
            f"{row['timeouts']} | {row['nonzero_returncodes']} | {len(row['malformed_records'])} | "
            f"{int(row['running_lock_present'])} |")
    (OUTPUT / "FINAL_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    cell_fields = ["campaign", "group", "records", "successes", "success_rate", "timeouts",
                   "nonzero_returncodes", "makespan_median", "makespan_p90", "makespan_max"]
    with (OUTPUT / "CELL_SUMMARY.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=cell_fields)
        writer.writeheader()
        for (campaign, group), records in sorted(grouped.items()):
            spans = sorted(value for record in records if (value := makespan(record)) is not None)
            p90 = None
            if spans:
                position = (len(spans) - 1) * 0.9
                lower, upper = math.floor(position), math.ceil(position)
                p90 = spans[lower] if lower == upper else (
                    spans[lower] * (upper - position) + spans[upper] * (position - lower))
            solved = sum(success(record) for record in records)
            writer.writerow({
                "campaign": campaign, "group": group, "records": len(records),
                "successes": solved, "success_rate": solved / len(records),
                "timeouts": sum(bool(record.get("timed_out")) for record in records),
                "nonzero_returncodes": sum(record.get("returncode", 0) not in (0, None)
                                           for record in records),
                "makespan_median": statistics.median(spans) if spans else None,
                "makespan_p90": p90, "makespan_max": max(spans) if spans else None,
            })
    print(OUTPUT / "FINAL_AUDIT.md")
    return 0 if all_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
