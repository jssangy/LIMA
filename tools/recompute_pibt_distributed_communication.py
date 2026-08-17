#!/usr/bin/env python3
"""Replay solved PIBT runs with paper-grounded distributed communication telemetry.

The reference pibt2 implementation is a centralized simulator, but the PIBT
paper maps recursive priority-inheritance calls and returned backtracking
outcomes to inter-agent messages.  The instrumented binary reports those exact
events without changing any planning decision.  For a complete logical actor
boundary model, this runner also includes one local state/priority announcement
and one selected-action announcement per active agent per timestep.

The resulting count is a broadcast-event lower bound: one broadcast is one
event regardless of how many nearby recipients receive it.  Network bytes and
recipient-weighted transmissions are intentionally reported separately from
this logical-message count.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import re
import statistics
import subprocess
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
INT_KEYS = (
    "pibt_comm_active_agent_steps",
    "pibt_comm_root_invocations",
    "pibt_comm_inheritance_requests",
    "pibt_comm_backtracking_responses",
    "pibt_comm_backtracking_valid",
    "pibt_comm_backtracking_invalid",
    "pibt_comm_max_propagation_depth",
    "pibt_comm_state_priority_announcements",
    "pibt_comm_decision_announcements",
    "pibt_comm_propagation_events",
    "pibt_comm_distributed_logical_events",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def parse_key_values(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in text.splitlines():
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        if key and " " not in key:
            result[key] = value.strip()
    return result


def parse_scalar(text: str, key: str) -> int:
    match = re.search(rf"(?:^|[,\s]){re.escape(key)}=\s*(-?\d+)", text, re.MULTILINE)
    if match is None:
        raise ValueError(f"missing scalar {key}")
    return int(match.group(1))


def strip_time_wrapper(command: list[str]) -> list[str]:
    if not command or command[0] != "/usr/bin/time":
        return list(command)
    output_index = command.index("-o")
    return list(command[output_index + 2 :])


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("1", "true", "yes")


def mean_var(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return statistics.fmean(values), statistics.variance(values) if len(values) > 1 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-campaign",
        type=Path,
        default=ROOT / "results/revision_final/oneshot_pibt_certified_step_v3",
    )
    parser.add_argument(
        "--batch-telemetry-campaign",
        type=Path,
        default=ROOT / "results/revision_final/oneshot_pibt_telemetry_backfill_v2",
    )
    parser.add_argument(
        "--instrumented-binary",
        type=Path,
        default=Path("/home/shlee/pibt-comm-instrument/build_comm/mapf"),
    )
    parser.add_argument(
        "--pibt-repo", type=Path, default=Path("/home/shlee/mapf-baselines/pibt2")
    )
    parser.add_argument(
        "--instrumented-source", type=Path, default=Path("/home/shlee/pibt-comm-instrument")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/revision_final/oneshot_pibt_distributed_comm_v1",
    )
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()

    source = args.source_campaign.resolve()
    batch = args.batch_telemetry_campaign.resolve()
    binary = args.instrumented_binary.resolve()
    pibt_repo = args.pibt_repo.resolve()
    instrumented_source = args.instrumented_source.resolve()
    output = args.output_dir.resolve()
    records_out = output / "records"
    logs_out = output / "logs"
    records_out.mkdir(parents=True, exist_ok=True)
    logs_out.mkdir(parents=True, exist_ok=True)

    source_manifest = source / "MANIFEST.json"
    if not source_manifest.is_file() or not binary.is_file():
        raise FileNotFoundError("source manifest or instrumented PIBT binary is missing")

    source_records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((source / "records").glob("*.json"))
    ]
    solved_records = [record for record in source_records if as_bool(record.get("solved"))]
    batch_by_tag: dict[str, dict] = {}
    for path in sorted((batch / "records").glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        batch_by_tag[str(value["tag"])] = value

    source_diff = subprocess.run(
        ["git", "-C", str(instrumented_source), "diff", "--binary"],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    fingerprint_payload = {
        "schema_version": 1,
        "source_manifest_sha256": sha256(source_manifest),
        "source_solved_records": len(solved_records),
        "instrumented_binary_sha256": sha256(binary),
        "instrumented_source_head": subprocess.run(
            ["git", "-C", str(instrumented_source), "rev-parse", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip(),
        "instrumented_source_diff_sha256": hashlib.sha256(source_diff).hexdigest(),
        "message_model": {
            "state_priority_announcement": "one broadcast per active agent-step",
            "decision_announcement": "one broadcast per active agent-step",
            "inheritance_request": "one event per recursive cross-agent PIBT call",
            "backtracking_response": "one event per returned recursive PIBT call",
            "recipient_weighting": "not applied; logical broadcast-event lower bound",
        },
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest_path = output / "MANIFEST.json"
    if manifest_path.is_file() and any(records_out.glob("*.json")):
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("experiment_fingerprint") != fingerprint:
            raise RuntimeError("output directory fingerprint mismatch")
    atomic_json(
        manifest_path,
        {
            **fingerprint_payload,
            "experiment_fingerprint": fingerprint,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_campaign": str(source),
            "batch_telemetry_campaign": str(batch),
            "instrumented_binary": str(binary),
        },
    )

    def run_one(record: dict) -> Path:
        tag = str(record["tag"])
        destination = records_out / f"{tag}_pibt_distributed_comm.json"
        if destination.is_file():
            existing = json.loads(destination.read_text(encoding="utf-8"))
            if existing.get("experiment_fingerprint") == fingerprint:
                return destination
        command = strip_time_wrapper(list(record["command"]))
        command[0] = str(binary)
        with tempfile.TemporaryDirectory(prefix="pibt-distributed-comm-") as directory:
            solution = Path(directory) / "solution.txt"
            if "--output" in command:
                command[command.index("--output") + 1] = str(solution)
            completed = subprocess.run(
                command,
                cwd=pibt_repo,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        (logs_out / f"{tag}.stdout.log").write_text(completed.stdout, encoding="utf-8")
        (logs_out / f"{tag}.stderr.log").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"instrumented PIBT failed for {tag}: {completed.returncode}")
        parsed = parse_key_values(completed.stdout)
        missing = [key for key in INT_KEYS if key not in parsed]
        if missing:
            raise RuntimeError(f"missing counters for {tag}: {missing}")
        counters = {key.removeprefix("pibt_comm_"): int(parsed[key]) for key in INT_KEYS}
        expected = record.get("result") or {}
        observed_scalars = {
            key: parse_scalar(completed.stdout, key)
            for key in ("solved", "makespan", "soc")
        }
        scalar_equivalence = {
            key: {
                "expected": int(expected[key]),
                "observed": observed_scalars[key],
                "match": int(expected[key]) == observed_scalars[key],
            }
            for key in ("solved", "makespan", "soc")
        }
        if not all(item["match"] for item in scalar_equivalence.values()):
            raise RuntimeError(f"scalar mismatch for {tag}: {scalar_equivalence}")
        active = counters["active_agent_steps"]
        batch_record = batch_by_tag.get(tag) or {}
        batch_comm = batch_record.get("communication") or {}
        batch_events = int(batch_comm.get("event_count") or 0)
        result = {
            "schema_version": 1,
            "experiment_fingerprint": fingerprint,
            "tag": tag,
            "map": record["map"],
            "target": record["target"],
            "scenario": int(record["scenario"]),
            "agents": int(record["agents"]),
            "source_record": str(source / "records" / f"{tag}_pibt.json"),
            "scalar_equivalence": scalar_equivalence,
            "communication": {
                "model": "PIBT native decentralized logical broadcast-event lower bound",
                "scope": "robot-to-robot; direct within two graph hops with multi-hop propagation",
                "counts": counters,
                "distributed_logical_events_per_active_agent_step": (
                    counters["distributed_logical_events"] / active if active else None
                ),
                "paper_propagation_events_per_active_agent_step": (
                    counters["propagation_events"] / active if active else None
                ),
                "inheritance_requests_per_active_agent_step": (
                    counters["inheritance_requests"] / active if active else None
                ),
                "batch_install_events_previous_model": batch_events,
                "batch_install_events_per_active_agent_step": (
                    batch_events / active if active else None
                ),
                "distributed_to_batch_event_ratio": (
                    counters["distributed_logical_events"] / batch_events
                    if batch_events else None
                ),
                "recipient_weighted_transmissions": None,
                "recipient_weighted_reason": (
                    "PIBT paper bounds direct communication by two-hop proximity but does not "
                    "fix a MAC/broadcast protocol; logical broadcasts are counted once"
                ),
            },
        }
        atomic_json(destination, result)
        return destination

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_one, record) for record in solved_records]
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            path = future.result()
            print(f"[{index}/{len(futures)}] {path.name}", flush=True)

    rows: list[dict[str, object]] = []
    for path in sorted(records_out.glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("experiment_fingerprint") != fingerprint:
            continue
        comm = value["communication"]
        counts = comm["counts"]
        rows.append(
            {
                "map": value["map"],
                "target": value["target"],
                "scenario": value["scenario"],
                "agents": value["agents"],
                "active_agent_steps": counts["active_agent_steps"],
                "distributed_logical_events": counts["distributed_logical_events"],
                "events_per_agent_step": comm["distributed_logical_events_per_active_agent_step"],
                "propagation_events_per_agent_step": comm["paper_propagation_events_per_active_agent_step"],
                "inheritance_requests_per_agent_step": comm["inheritance_requests_per_active_agent_step"],
                "max_propagation_depth": counts["max_propagation_depth"],
                "batch_events_per_agent_step": comm["batch_install_events_per_active_agent_step"],
                "distributed_to_batch_ratio": comm["distributed_to_batch_event_ratio"],
            }
        )
    if len(rows) != len(solved_records):
        raise RuntimeError(f"record count mismatch: {len(rows)} != {len(solved_records)}")

    with (output / "records.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["map"]), str(row["target"]))].append(row)
    summary_rows: list[dict[str, object]] = []
    metric_names = (
        "events_per_agent_step",
        "propagation_events_per_agent_step",
        "inheritance_requests_per_agent_step",
        "max_propagation_depth",
        "batch_events_per_agent_step",
        "distributed_to_batch_ratio",
    )
    for (map_name, target), selected in sorted(grouped.items()):
        row: dict[str, object] = {"map": map_name, "target": target, "n": len(selected)}
        for metric in metric_names:
            values = [float(item[metric]) for item in selected if item[metric] is not None]
            mean, variance = mean_var(values)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_variance"] = variance
        summary_rows.append(row)
    all_summary: dict[str, object] = {"map": "ALL", "target": "ALL", "n": len(rows)}
    for metric in metric_names:
        values = [float(item[metric]) for item in rows if item[metric] is not None]
        mean, variance = mean_var(values)
        all_summary[f"{metric}_mean"] = mean
        all_summary[f"{metric}_variance"] = variance
    summary_rows.append(all_summary)

    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = [
        "# PIBT Native Distributed Communication Recalculation",
        "",
        f"- Solved runs replayed: {len(rows)}",
        "- Scalar equivalence: solved, makespan, and SOC matched for every replay.",
        "- Event model: one state/priority broadcast and one chosen-action broadcast per active agent-step, plus exact priority-inheritance requests and returned backtracking responses.",
        "- Counts are logical broadcast events, not recipient-weighted radio transmissions.",
        "",
        "| Map | Load | n | Distributed events / agent-step mean (var) | Propagation / agent-step mean (var) | Max depth mean (var) | Old batch events / agent-step mean (var) | New / old mean (var) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        def shown(metric: str) -> str:
            mean = row.get(f"{metric}_mean")
            variance = row.get(f"{metric}_variance")
            return "—" if mean is None else f"{float(mean):.6f} ({float(variance):.6f})"

        lines.append(
            f"| {row['map']} | {row['target']} | {row['n']} | "
            f"{shown('events_per_agent_step')} | {shown('propagation_events_per_agent_step')} | "
            f"{shown('max_propagation_depth')} | {shown('batch_events_per_agent_step')} | "
            f"{shown('distributed_to_batch_ratio')} |"
        )
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output / "SUMMARY.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
