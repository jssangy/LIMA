#!/usr/bin/env python3
"""Run the staged Admission/Recirculation information-axis evaluation.

The runner is deliberately step-bounded and resumable.  It uses the frozen
capacity-certified task inputs, the current ``lima-default`` profile, and
changes only opt-in information-axis flags.  Each cell is an atomic JSON
record; later stages deterministically select candidates from the preceding
stage using the paper's ranking order.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gzip
import hashlib
import json
import math
import os
import platform
import shutil
import socket
import statistics
import subprocess
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "results/infoaxis"
DEFAULT_INPUT = ROOT / "results/revision_final/certified_inputs_v3/MANIFEST.json"
DEFAULT_BINARY = ROOT / "build_infoaxis/lima"


@dataclass(frozen=True)
class Variant:
    name: str
    family: str
    flags: tuple[str, ...] = ()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        for token in line.split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            fields[key] = value
    return fields


def parse_resource(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    parsed = parse_fields(path.read_text(encoding="utf-8", errors="replace"))
    result: dict[str, float] = {}
    for key, value in parsed.items():
        try:
            result[key] = float(value)
        except ValueError:
            continue
    return result


def full_variants() -> list[Variant]:
    variants = [Variant("base", "base")]
    variants += [
        Variant("a1_hard", "A1", ("--admit-lookahead", "hard")),
        *[
            Variant(f"a1_thresh{x}", "A1", (
                "--admit-lookahead", "thresh", "--admit-lookahead-param", str(x)))
            for x in (1, 2, 3)
        ],
        *[
            Variant(f"a1_ratio{label}", "A1", (
                "--admit-lookahead", "ratio", "--admit-lookahead-param", str(value)))
            for label, value in (("85", 0.85), ("95", 0.95), ("100", 1.0))
        ],
        *[
            Variant(f"a1_diff{label}", "A1", (
                "--admit-lookahead", "diff", "--admit-lookahead-param", str(value)))
            for label, value in (("m25", -0.25), ("0", 0.0), ("p25", 0.25))
        ],
    ]
    variants += [
        *[
            Variant(f"a2_nbmax{label}", "A2", (
                "--aimd-signal", "nbmax", "--aimd-signal-param", str(value)))
            for label, value in (("85", 0.85), ("95", 0.95))
        ],
        *[
            Variant(f"a2_nbmean{label}", "A2", (
                "--aimd-signal", "nbmean", "--aimd-signal-param", str(value)))
            for label, value in (("85", 0.85), ("95", 0.95))
        ],
        *[
            Variant(f"a2_trend{x}", "A2", (
                "--aimd-signal", "trend", "--aimd-signal-param", str(x)))
            for x in (2, 3)
        ],
    ]
    variants += [
        Variant("a3_equal", "A3", ("--admit-credit", "equal")),
        Variant("a3_demand", "A3", ("--admit-credit", "demand")),
        *[
            Variant(f"a3_drr{label}", "A3", (
                "--admit-credit", "drr", "--admit-credit-param", str(value)))
            for label, value in (("05", 0.5), ("10", 1.0), ("20", 2.0))
        ],
    ]
    for mode_name, mode in (
        ("detect", "detect"), ("slack", "break-slack"),
        ("longarm", "break-longarm"),
    ):
        for ttl in (4, 6, 8):
            for age in (1, 3, 5):
                variants.append(Variant(
                    f"r1_{mode_name}_t{ttl}_a{age}", "R1", (
                        "--recirc-probe", mode,
                        "--recirc-probe-ttl", str(ttl),
                        "--recirc-probe-age", str(age),
                    )))
    variants += [
        Variant("r2_id", "R2", ("--recirc-exclusive", "id")),
        Variant("r2_age", "R2", ("--recirc-exclusive", "age")),
        Variant("r2_reserve", "R2", ("--recirc-exclusive", "reserve")),
        Variant("r3_l6", "R3", ("--recirc-cycle-max", "6")),
        Variant("r3_l8", "R3", ("--recirc-cycle-max", "8")),
    ]
    return variants


def quick_variants() -> list[Variant]:
    """Small boundary screen after the full single-cell sweep.

    Keep one representative per implemented information family plus the
    combinations that can reveal the most important interaction.  R3 is
    omitted because the prerequisite audit found a four-cycle at every
    managed intersection on all three paper maps.
    """
    a1 = Variant("a1_ratio95", "A1", (
        "--admit-lookahead", "ratio", "--admit-lookahead-param", "0.95"))
    a2 = Variant("a2_trend3", "A2", (
        "--aimd-signal", "trend", "--aimd-signal-param", "3"))
    a3 = Variant("a3_drr05", "A3", (
        "--admit-credit", "drr", "--admit-credit-param", "0.5"))
    r1 = Variant("r1_slack_t4_a1", "R1", (
        "--recirc-probe", "break-slack", "--recirc-probe-ttl", "4",
        "--recirc-probe-age", "1"))
    r2 = Variant("r2_age", "R2", ("--recirc-exclusive", "age"))
    return [
        Variant("base", "base"), a1, a2, a3, r1, r2,
        combine("admission_stack", "stack", [a1, a2, a3]),
        combine("cross_best", "stack", [a1, r1]),
        combine("full_info_stack", "stack", [a1, a2, a3, r1, r2]),
    ]


def load_summary(root: Path, stage: int) -> dict:
    path = root / f"stage{stage}" / "summary/summary.json"
    if not path.is_file():
        raise RuntimeError(f"preceding stage summary is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def variants_from_summary(summary: dict) -> dict[str, Variant]:
    return {
        row["variant"]: Variant(row["variant"], row["family"], tuple(row["flags"]))
        for row in summary["variants"]
    }


def ranked_rows(summary: dict, family: str | None = None) -> list[dict]:
    rows = [row for row in summary["variants"] if row["variant"] != "base"]
    if family is not None:
        rows = [row for row in rows if row["family"] == family]
    return sorted(rows, key=lambda row: tuple(row["rank_key"]))


def combine(name: str, family: str, variants: list[Variant]) -> Variant:
    flags: list[str] = []
    for variant in variants:
        flags.extend(variant.flags)
    return Variant(name, family, tuple(flags))


def stage_variants(output_root: Path, stage: int) -> list[Variant]:
    if stage == -1:
        return [
            Variant("ack_alpha25", "baseline", ("--gate-param2", "0.25")),
            Variant("ack_alpha50", "baseline", ("--gate-param2", "0.50")),
        ]
    if stage == 0:
        return full_variants()
    if stage == 1:
        previous = load_summary(output_root, 0)
        allowed = {
            row["variant"] for row in previous["variants"]
            if row["variant"] == "base"
            or (row["errors"] == 0 and row["validation_failures"] == 0
                and row["success_cells"] == row["cells"])
        }
        return [variant for variant in full_variants() if variant.name in allowed]
    if stage == 2:
        previous = load_summary(output_root, 1)
        available = variants_from_summary(previous)
        selected = [Variant("base", "base")]
        for family in ("A1", "A2", "A3", "R1", "R2", "R3"):
            selected.extend(available[row["variant"]] for row in ranked_rows(previous, family)[:2])
        return selected
    if stage == 3:
        previous = load_summary(output_root, 2)
        available = variants_from_summary(previous)
        best = {
            family: available[ranked_rows(previous, family)[0]["variant"]]
            for family in ("A1", "A2", "A3", "R1", "R2", "R3")
        }
        best_admission = min(
            (row for family in ("A1", "A2", "A3") for row in ranked_rows(previous, family)[:1]),
            key=lambda row: tuple(row["rank_key"]),
        )
        best_recirc = min(
            (row for family in ("R1", "R2", "R3") for row in ranked_rows(previous, family)[:1]),
            key=lambda row: tuple(row["rank_key"]),
        )
        singles = [best[key] for key in ("A1", "A2", "A3", "R1", "R2", "R3")]
        return [
            Variant("base", "base"), *singles,
            combine("pair_a1_a3", "pair", [best["A1"], best["A3"]]),
            combine("pair_r1_r2", "pair", [best["R1"], best["R2"]]),
            combine("admission_stack", "stack", [best["A1"], best["A2"], best["A3"]]),
            combine("recirc_stack", "stack", [best["R1"], best["R2"], best["R3"]]),
            combine("cross_best", "stack", [
                available[best_admission["variant"]], available[best_recirc["variant"]]]),
            combine("full_stack", "stack", singles),
        ]
    if stage == 4:
        previous = load_summary(output_root, 3)
        available = variants_from_summary(previous)
        return [Variant("base", "base"), *[
            available[row["variant"]] for row in ranked_rows(previous)[:3]
        ]]
    if stage == 5:
        previous = load_summary(output_root, 4)
        available = variants_from_summary(previous)
        winner = ranked_rows(previous)[0]
        return [Variant("base", "base"), available[winner["variant"]]]
    raise ValueError(f"unsupported stage: {stage}")


def stage_specs(stage: int) -> tuple[int, list[tuple[str, str, int]]]:
    if stage == -1:
        return 7000, [
            ("cross_3030", "d50", 0), ("cross_3030", "d50", 1),
            ("warehouse_10_20", "d50", 0), ("warehouse_10_20", "d50", 1),
            ("warehouse_20_40", "d30", 0), ("warehouse_20_40", "d30", 1),
        ]
    if stage == 0:
        return 1000, [("warehouse_10_20", "d10", 0)]
    if stage == 1:
        return 7000, [("warehouse_10_20", "d50", 0)]
    if stage in (2, 3):
        return 7000, [
            ("cross_3030", "d50", 0), ("cross_3030", "d50", 1),
            ("warehouse_10_20", "d50", 0), ("warehouse_10_20", "d50", 1),
            ("warehouse_20_40", "d30", 0), ("warehouse_20_40", "d30", 1),
        ]
    if stage == 4:
        return 7000, [
            (map_name, f"d{density}", scenario)
            for map_name in ("cross_3030", "warehouse_10_20", "warehouse_20_40")
            for density in (10, 20, 30, 40, 50)
            for scenario in (0, 1)
        ]
    if stage == 5:
        return 7000, [
            ("cross_3030", "d50", 0),
            ("warehouse_20_40", "d30", 0),
        ]
    raise ValueError(f"unsupported stage: {stage}")


def cell_from_certificate(input_root: Path, map_name: str, target: str, scenario: int) -> dict:
    certificate = input_root / f"certificates/{map_name}/{target}_s{scenario}.json"
    if not certificate.is_file():
        raise FileNotFoundError(certificate)
    data = json.loads(certificate.read_text(encoding="utf-8"))
    map_file = ROOT / data["map_file"]
    scenario_file = ROOT / data["scenario_file"]
    return {
        "tag": f"{map_name}_{target}_a{data['agents']}_s{scenario}",
        "map": map_name,
        "target": target,
        "density_percent": int(target.removeprefix("d")),
        "scenario": scenario,
        "agents": int(data["agents"]),
        "map_file": str(map_file.relative_to(ROOT)),
        "scenario_file": str(scenario_file.relative_to(ROOT)),
        "certificate_file": str(certificate.relative_to(ROOT)),
        "map_sha256": sha256(map_file),
        "scenario_sha256": sha256(scenario_file),
        "certificate_sha256": sha256(certificate),
    }


def controller_metrics(path: Path, steps: int) -> dict:
    source = path / "controller_information_events.csv"
    result = {
        "events": {}, "activations": 0, "messages": 0, "bytes": 0,
        "max_information_hops": 0, "max_message_hops": 0,
        "max_delay_steps": 0,
        "messages_per_step": 0.0, "link_messages_per_step": {},
    }
    if not source.is_file():
        return result
    links: dict[str, int] = defaultdict(int)
    with source.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            count = int(row["count"])
            payload = int(row["bytes"])
            key = f"{row['mechanism']}:{row['event']}"
            result["events"][key] = result["events"].get(key, 0) + count
            result["activations"] += count
            hops = int(row["hops"])
            result["max_information_hops"] = max(
                result["max_information_hops"], hops)
            result["max_delay_steps"] = max(
                result["max_delay_steps"], int(row["delay_steps"]))
            if payload > 0:
                result["messages"] += count
                result["bytes"] += count * payload
                result["max_message_hops"] = max(
                    result["max_message_hops"], hops)
                links[f"{row['source']}->{row['target']}"] += count
    denominator = max(1, steps)
    result["messages_per_step"] = result["messages"] / denominator
    rates = [value / denominator for value in links.values()]
    if rates:
        result["link_messages_per_step"] = {
            "count": len(rates), "mean": statistics.fmean(rates),
            "max": max(rates), "variance": statistics.pvariance(rates),
        }
    return result


def path_validation(path: Path) -> dict:
    source = path / "path_validation.csv"
    if not source.is_file():
        return {"ok": False, "reason": "missing path_validation.csv"}
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        return {"ok": False, "reason": f"expected one row, found {len(rows)}"}
    row = rows[0]
    return {
        "ok": row.get("ok") == "1",
        **{key: int(row[key]) for key in (
            "steps_observed", "invalid_moves", "vertex_conflicts",
            "edge_conflicts", "completed_goal_mismatches")},
    }


def gzip_metrics(path: Path) -> list[dict]:
    files: list[dict] = []
    if not path.is_dir():
        return files
    for source in sorted(path.glob("*.csv")):
        target = source.with_suffix(source.suffix + ".gz")
        with source.open("rb") as incoming, gzip.open(target, "wb", compresslevel=6) as outgoing:
            shutil.copyfileobj(incoming, outgoing)
        source.unlink()
        files.append({
            "file": target.name, "sha256": sha256(target), "bytes": target.stat().st_size,
        })
    return files


def trace_validation(trace: Path) -> dict:
    verification = subprocess.run(
        ["python3", str(ROOT / "tools/verify_trace.py"), str(trace)],
        cwd=ROOT, text=True, capture_output=True,
    )
    try:
        result = json.loads(verification.stdout)
    except json.JSONDecodeError:
        result = {"ok": False, "stdout": verification.stdout, "stderr": verification.stderr}
    result["returncode"] = verification.returncode
    result["sha256"] = sha256(trace) if trace.is_file() else None
    return result


def rank_key(row: dict) -> list[float]:
    makespan = row["makespan_median"]
    if makespan is None:
        makespan = 1e30
    return [
        row["errors"] + row["validation_failures"],
        -row["success_cells"], row["residual_agents"],
        -row["agent_completion_ratio"], makespan,
        row["moves_median"] if row["moves_median"] is not None else 1e30,
        row["waits_median"] if row["waits_median"] is not None else 1e30,
    ]


def summarize(stage_root: Path, stage: int, variants: list[Variant], cells: list[dict]) -> dict:
    cell_rows: list[dict] = []
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for variant in variants:
        for record_path in sorted((stage_root / "variants" / variant.name / "records").glob("*.json")):
            record = json.loads(record_path.read_text(encoding="utf-8"))
            result = record.get("result", {})
            completed_text = result.get("completed", f"0/{record['agents']}")
            try:
                completed, total = map(int, completed_text.split("/", 1))
            except ValueError:
                completed, total = 0, int(record["agents"])
            validation = record.get("path_validation", {})
            success = bool(
                record.get("returncode") == 0
                and result.get("status") == "completed"
                and completed == total and validation.get("ok") is True
                and record.get("trace_validation", {}).get("ok", True) is True
            )
            row = {
                "stage": stage, "variant": variant.name, "family": variant.family,
                "flags": list(variant.flags), "tag": record["tag"],
                "map": record["map"], "density_percent": record["density_percent"],
                "scenario": record["scenario"], "agents": record["agents"],
                "returncode": record.get("returncode"),
                "status": result.get("status", "error"), "success": success,
                "completed": completed, "total": total, "residual": total - completed,
                "agent_completion_ratio": completed / total if total else 0.0,
                "steps": int(result.get("steps", 0) or 0),
                "moves": int(result.get("moves", 0) or 0),
                "waits": int(result.get("waits", 0) or 0),
                "deadlocks": int(result.get("deadlocks", 0) or 0),
                "path_validation_ok": validation.get("ok") is True,
                "trace_validation_ok": record.get("trace_validation", {}).get("ok", True) is True,
                "runner_wall_seconds": record.get("runner_wall_seconds", 0.0),
                "controller_messages": record.get("controller_information", {}).get("messages", 0),
                "controller_bytes": record.get("controller_information", {}).get("bytes", 0),
            }
            cell_rows.append(row)
            by_variant[variant.name].append(row)

    variant_rows: list[dict] = []
    variant_lookup = {variant.name: variant for variant in variants}
    for name, rows in by_variant.items():
        successful = [row for row in rows if row["success"]]
        total_agents = sum(row["total"] for row in rows)
        variant = variant_lookup[name]
        aggregate = {
            "variant": name, "family": variant.family, "flags": list(variant.flags),
            "cells": len(rows), "success_cells": len(successful),
            "errors": sum(row["returncode"] not in (0, 2) for row in rows),
            "validation_failures": sum(
                not row["path_validation_ok"] or not row["trace_validation_ok"] for row in rows),
            "completed_agents": sum(row["completed"] for row in rows),
            "total_agents": total_agents,
            "residual_agents": sum(row["residual"] for row in rows),
            "agent_completion_ratio": (
                sum(row["completed"] for row in rows) / total_agents if total_agents else 0.0),
            "makespan_median": statistics.median(
                row["steps"] for row in successful) if successful else None,
            "moves_median": statistics.median(
                row["moves"] for row in successful) if successful else None,
            "waits_median": statistics.median(
                row["waits"] for row in successful) if successful else None,
            "controller_messages": sum(row["controller_messages"] for row in rows),
            "controller_bytes": sum(row["controller_bytes"] for row in rows),
        }
        aggregate["rank_key"] = rank_key(aggregate)
        variant_rows.append(aggregate)
    variant_rows.sort(key=lambda row: tuple(row["rank_key"]))

    summary_dir = stage_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    if cell_rows:
        with (summary_dir / "cells.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=[
                key for key in cell_rows[0] if key != "flags"])
            writer.writeheader()
            writer.writerows({key: value for key, value in row.items() if key != "flags"}
                             for row in cell_rows)
        csv_rows = [{
            key: (json.dumps(value) if key in ("flags", "rank_key") else value)
            for key, value in row.items()
        } for row in variant_rows]
        with (summary_dir / "variants.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)

    summary = {
        "stage": stage, "created_utc": datetime.now(timezone.utc).isoformat(),
        "records": len(cell_rows), "expected_records": len(variants) * len(cells),
        "complete": len(cell_rows) == len(variants) * len(cells),
        "variants": variant_rows,
    }
    atomic_json(summary_dir / "summary.json", summary)
    lines = [
        f"# Information-axis Stage {stage}", "",
        f"- Records: {summary['records']}/{summary['expected_records']}",
        f"- Complete: {'yes' if summary['complete'] else 'no'}", "",
        "| rank | variant | family | completed | residual | ACR | makespan median | moves median | messages | validation failures |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(variant_rows, 1):
        makespan = "-" if row["makespan_median"] is None else f"{row['makespan_median']:.0f}"
        moves = "-" if row["moves_median"] is None else f"{row['moves_median']:.0f}"
        lines.append(
            f"| {index} | {row['variant']} | {row['family']} | "
            f"{row['success_cells']}/{row['cells']} | {row['residual_agents']} | "
            f"{row['agent_completion_ratio']:.6f} | {makespan} | {moves} | "
            f"{row['controller_messages']} | {row['validation_failures']} |"
        )
    (summary_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def update_root_report(output_root: Path) -> None:
    lines = [
        "# LIMA controller information-axis evaluation", "",
        "The reference is `lima-default` profile v4: acknowledged AIMD "
        "(beta=0.25, additive ACK recovery=0.25), beam-complete Marshalling Solver, "
        "SWR Route Planner, and composite Recirculation Controller.", "",
        "## Locality contract", "",
        "| Mechanism | Dynamic information | Radius | Delay |",
        "|---|---|---:|---:|",
        "| A1 | next-hop admission availability/occupancy | 1 hop | one controller cycle |",
        "| A2 | adjacent occupancy aggregate | 1 hop | one controller cycle |",
        "| A3 | directional demand and advertised credits | 1 hop | one controller cycle |",
        "| R1 | wait-for probe forwarded edge by edge | one hop/message | cycle length |",
        "| R2 | neighboring recirculation advertisement | 1 hop | current arbitration cycle |",
        "| R3 | static topology only | bounded static radius | none |",
        "", "## Stages", "",
    ]
    for stage in range(6):
        summary_path = output_root / f"stage{stage}/summary/summary.json"
        if not summary_path.is_file():
            lines.append(f"- Stage {stage}: pending")
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        winner = summary["variants"][0]["variant"] if summary["variants"] else "-"
        lines.append(
            f"- Stage {stage}: {summary['records']}/{summary['expected_records']} records; "
            f"winner `{winner}`; complete={'yes' if summary['complete'] else 'no'}")
    lines += [
        "", "Stage ranking is watchdog/error, completed cells, residual agents, "
        "agent completion ratio, makespan, moves, then waits. Wall time is retained "
        "only as operational metadata.", "",
    ]
    (output_root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def find_reusable(output_root: Path, stage: int, variant: Variant, cell: dict,
                  binary_sha: str, max_steps: int) -> Path | None:
    if stage in (0, 5):
        return None
    for previous in range(stage - 1, -1, -1):
        variant_root = output_root / f"stage{previous}/variants"
        if not variant_root.is_dir():
            continue
        for record_path in variant_root.glob(f"*/records/{cell['tag']}.json"):
            try:
                record = json.loads(record_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (
                tuple(record.get("variant_flags", [])) == variant.flags
                and record.get("binary_sha256") == binary_sha
                and record.get("max_steps") == max_steps
                and record.get("scenario_sha256") == cell["scenario_sha256"]
                and record.get("certificate_sha256") == cell["certificate_sha256"]
            ):
                return record_path
    return None


def run_stage(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    input_manifest = args.input_manifest.resolve()
    input_root = input_manifest.parent
    binary = args.binary.resolve()
    runner = Path(__file__).resolve()
    for path in (input_manifest, binary):
        if not path.is_file():
            raise FileNotFoundError(path)

    variants = quick_variants() if args.quick else stage_variants(output_root, args.stage)
    max_steps, specs = stage_specs(args.stage)
    cells = [cell_from_certificate(input_root, *spec) for spec in specs]
    stage_root = output_root / f"stage{args.stage}"
    binary_sha = sha256(binary)
    version = subprocess.run(
        [str(binary), "--version"], cwd=ROOT, text=True,
        capture_output=True, check=True).stdout.strip()
    fingerprint_payload = {
        "schema_version": 1, "stage": args.stage,
        "runner_sha256": sha256(runner), "binary_sha256": binary_sha,
        "binary_version": version, "input_manifest_sha256": sha256(input_manifest),
        "max_steps": max_steps, "quick": args.quick,
        "baseline_alpha": args.baseline_alpha, "variants": [
            {"name": variant.name, "family": variant.family, "flags": list(variant.flags)}
            for variant in variants],
        "cells": cells,
    }
    fingerprint = json_hash(fingerprint_payload)
    manifest_path = stage_root / "MANIFEST.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("experiment_fingerprint") != fingerprint:
            raise RuntimeError(f"stage output contains a different fingerprint: {stage_root}")
    if args.dry_run:
        print(json.dumps({
            "stage": args.stage, "variants": len(variants), "cells": len(cells),
            "jobs": len(variants) * len(cells), "max_steps": max_steps,
            "variant_names": [variant.name for variant in variants],
            "cell_tags": [cell["tag"] for cell in cells],
        }, indent=2))
        return 0

    stage_root.mkdir(parents=True, exist_ok=True)
    atomic_json(manifest_path, {
        **fingerprint_payload, "experiment_fingerprint": fingerprint,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "jobs_concurrency": args.jobs, "cpu_list": args.cpu_list,
        "host": {"hostname": socket.gethostname(), "platform": platform.platform(),
                 "cpu_count": os.cpu_count()},
    })
    lock = stage_root / ".RUNNING"
    if lock.exists():
        raise RuntimeError(f"stage lock exists: {lock}")
    lock.write_text(
        f"pid={os.getpid()}\nstarted_utc={datetime.now(timezone.utc).isoformat()}\n",
        encoding="utf-8")

    jobs = [(variant, cell) for variant in variants for cell in cells]

    def run_one(variant: Variant, cell: dict) -> tuple[str, str, str]:
        variant_root = stage_root / "variants" / variant.name
        records = variant_root / "records"
        logs = variant_root / "logs"
        resources = variant_root / "resources"
        metrics_root = variant_root / "metrics"
        traces = variant_root / "traces"
        for path in (records, logs, resources, metrics_root):
            path.mkdir(parents=True, exist_ok=True)
        record_path = records / f"{cell['tag']}.json"
        if record_path.is_file() and not args.rerun:
            record = json.loads(record_path.read_text(encoding="utf-8"))
            return variant.name, cell["tag"], "skipped" if record.get("returncode") in (0, 2) else "skipped_error"

        reusable = None if args.rerun else find_reusable(
            output_root, args.stage, variant, cell, binary_sha, max_steps)
        if reusable is not None:
            record = json.loads(reusable.read_text(encoding="utf-8"))
            record.update({
                "stage": args.stage, "variant": variant.name, "family": variant.family,
                "experiment_fingerprint": fingerprint,
                "reused_from": str(reusable.relative_to(ROOT)),
            })
            atomic_json(record_path, record)
            return variant.name, cell["tag"], "reused"

        log_path = logs / f"{cell['tag']}.log"
        resource_path = resources / f"{cell['tag']}.txt"
        metrics_path = metrics_root / cell["tag"]
        metrics_path.mkdir(parents=True, exist_ok=True)
        command = [
            "taskset", "-c", args.cpu_list,
            "/usr/bin/time", "-f",
            "max_rss_kb=%M\nuser_seconds=%U\nsystem_seconds=%S\nelapsed_seconds=%e",
            "-o", str(resource_path), str(binary),
            "--profile", "lima-default", "--mode", "solve",
            "--map", cell["map_file"], "--scenario", cell["scenario_file"],
            "--agents", str(cell["agents"]), "--seed", str(cell["scenario"]),
            "--max-steps", str(max_steps), "--stall-threshold", str(max_steps + 1),
            "--goal-behavior", "disappear", "--no-trace",
            "--metrics", str(metrics_path),
            "--gate-param2", str(args.baseline_alpha), *variant.flags,
        ]
        trace_path = None
        if args.stage in (0, 5):
            traces.mkdir(parents=True, exist_ok=True)
            trace_path = traces / f"{cell['tag']}.jsonl"
            command += ["--trace-jsonl", str(trace_path)]
        started = time.time()
        proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        elapsed = time.time() - started
        result = parse_fields(proc.stdout)
        steps = int(result.get("steps", 0) or 0)
        validation = path_validation(metrics_path)
        information = controller_metrics(metrics_path, steps)
        trace_result = ({"ok": False, "reason": "missing trace"}
                        if trace_path is not None else {})
        trace_artifact = None
        if trace_path is not None and trace_path.is_file():
            trace_result = trace_validation(trace_path)
            if args.stage == 5:
                compressed = trace_path.with_suffix(trace_path.suffix + ".gz")
                with trace_path.open("rb") as incoming, gzip.open(
                        compressed, "wb", compresslevel=6) as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
                trace_path.unlink()
                trace_artifact = {
                    "file": str(compressed.relative_to(ROOT)),
                    "sha256": sha256(compressed), "bytes": compressed.stat().st_size,
                }
            else:
                trace_path.unlink()
        metric_artifacts = gzip_metrics(metrics_path)
        log_path.write_text(
            proc.stdout + ("\n[stderr]\n" + proc.stderr if proc.stderr else ""),
            encoding="utf-8")
        record = {
            **cell, "stage": args.stage, "variant": variant.name,
            "family": variant.family, "variant_flags": list(variant.flags),
            "binary": str(binary.relative_to(ROOT)), "binary_sha256": binary_sha,
            "binary_version": version, "max_steps": max_steps,
            "returncode": proc.returncode, "result": result,
            "runner_wall_seconds": elapsed, "resource": parse_resource(resource_path),
            "path_validation": validation, "trace_validation": trace_result,
            "trace_artifact": trace_artifact,
            "controller_information": information,
            "metric_artifacts": metric_artifacts,
            "command": command, "log": str(log_path.relative_to(ROOT)),
            "experiment_fingerprint": fingerprint,
        }
        atomic_json(record_path, record)
        status = "completed" if (
            proc.returncode == 0 and result.get("status") == "completed"
            and validation.get("ok") is True and trace_result.get("ok", True) is True
        ) else result.get("status", f"returncode_{proc.returncode}")
        return variant.name, cell["tag"], status

    try:
        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(run_one, *job) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                variant, tag, status = future.result()
                completed += 1
                print(f"[{completed:3d}/{len(jobs):3d}] {status:14s} {variant:28s} {tag}", flush=True)
        summary = summarize(stage_root, args.stage, variants, cells)
        update_root_report(output_root)
        if not summary["complete"]:
            raise RuntimeError("stage record set is incomplete")
    finally:
        lock.unlink(missing_ok=True)
    print(stage_root / "summary/REPORT.md")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=range(-1, 6), required=True,
                        help="-1 calibrates acknowledged AIMD alpha; 0-5 run the staged screen")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--cpu-list", default="0-7")
    parser.add_argument("--baseline-alpha", type=float, default=0.25)
    parser.add_argument("--quick", action="store_true",
                        help="run the compact representative screen for the selected stage specs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.jobs <= (os.cpu_count() or 1):
        parser.error("--jobs must be in [1, available CPU count]")
    return run_stage(args)


if __name__ == "__main__":
    raise SystemExit(main())
