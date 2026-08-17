#!/usr/bin/env python3
"""Analyze the information-axis prerequisites without running LIMA.

P1 reads already completed 100k-step records.  P2 reproduces the shipped
intersection and cycle enumeration on static maps.  P3 audits whether the
legacy telemetry can answer the requested saturated-downstream admission
question; it deliberately reports the metric as unavailable rather than
substituting a different counter.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORDS = ROOT / "results/revision_final/oneshot_lima_certified_step_v4_optimized/records"
DEFAULT_MAPS = ROOT / "results/revision_final/certified_inputs_v3/maps"
DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))


def percentile(values: list[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = fraction * (len(ordered) - 1)
    lower = int(index)
    upper = min(len(ordered) - 1, lower + 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def p1_completion_tail(records_dir: Path) -> dict:
    rows: list[dict] = []
    for path in sorted(records_dir.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("algorithm") != "lima" or record.get("target") not in {"d40", "d50"}:
            continue
        result = record.get("result", {})
        if result.get("status") != "completed":
            continue
        rows.append({
            "map": record["map"], "density": record["target"],
            "scenario": record["scenario"], "steps": int(result["steps"]),
        })

    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in rows:
        groups[(row["map"], row["density"])].append(row["steps"])

    def summarize(values: list[int]) -> dict:
        tail = [value for value in values if value > 5000]
        return {
            "records": len(values), "min_steps": min(values),
            "mean_steps": statistics.fmean(values),
            "median_steps": statistics.median(values), "max_steps": max(values),
            "over_5000": len(tail),
            "over_5000_fraction": len(tail) / len(values),
            "tail_steps": sorted(tail),
            "tail_excess_mean": statistics.fmean(value - 5000 for value in tail) if tail else 0.0,
            "tail_p50": percentile(tail, 0.50), "tail_p90": percentile(tail, 0.90),
            "tail_p99": percentile(tail, 0.99),
            "within_6000": sum(value <= 6000 for value in tail),
            "within_10000": sum(value <= 10000 for value in tail),
            "over_30000": sum(value > 30000 for value in tail),
        }

    return {
        "source": str(records_dir.relative_to(ROOT)),
        "semantics": "legacy profile-v1 simple-window AIMD; diagnostic only",
        "groups": {
            f"{map_name}:{density}": summarize(values)
            for (map_name, density), values in sorted(groups.items())
        },
        "overall": summarize([row["steps"] for row in rows]),
    }


def load_map(path: Path) -> tuple[list[str], set[tuple[int, int]], set[tuple[int, int]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    marker = lines.index("map")
    grid = lines[marker + 1:]
    free = set()
    goals = set()
    for y, row in enumerate(grid):
        for x, value in enumerate(row):
            if value in ".SEG":
                free.add((x, y))
            if value in "SG":
                goals.add((x, y))
    return grid, free, goals


def build_topology(path: Path) -> list[dict]:
    _, free, goals = load_map(path)

    def wall(cell: tuple[int, int]) -> bool:
        return cell not in free

    def center(cell: tuple[int, int]) -> bool:
        x, y = cell
        open_neighbors = sum((x + dx, y + dy) in free for dx, dy in DELTAS)
        return open_neighbors >= 3 and all(
            wall((x + dx, y + dy)) for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1)))

    intersections: list[dict] = []
    by_center: dict[tuple[int, int], int] = {}
    for cell in sorted(free, key=lambda value: (value[1], value[0])):
        if not center(cell):
            continue
        arms: list[list[tuple[int, int]]] = []
        for dx, dy in DELTAS:
            arm = []
            x, y = cell[0] + dx, cell[1] + dy
            while (x, y) in free:
                if (x, y) in goals:
                    break
                corridor = (
                    wall((x - 1, y)) and wall((x + 1, y)) if dy
                    else wall((x, y - 1)) and wall((x, y + 1)))
                if not corridor:
                    break
                arm.append((x, y))
                x, y = x + dx, y + dy
            arms.append(arm)
        if sum(bool(arm) for arm in arms) < 3:
            continue
        index = len(intersections)
        by_center[cell] = index
        intersections.append({"center": cell, "arms": arms, "neighbors": [-1] * 4})

    for item in intersections:
        for direction, arm in enumerate(item["arms"]):
            if not arm:
                continue
            dx, dy = DELTAS[direction]
            expected = (arm[-1][0] + dx, arm[-1][1] + dy)
            item["neighbors"][direction] = by_center.get(expected, -1)
    return intersections


def cycle_lengths(intersections: list[dict], source: int, maximum: int = 8) -> set[int]:
    lengths: set[int] = set()
    for b in intersections[source]["neighbors"]:
        if b < 0:
            continue
        around = [node for node in intersections[b]["neighbors"] if node >= 0 and node != source]
        for i, c in enumerate(around):
            for e in around[i + 1:]:
                common = set(node for node in intersections[c]["neighbors"] if node >= 0)
                common.intersection_update(node for node in intersections[e]["neighbors"] if node >= 0)
                if any(node not in {b, c, e} for node in common):
                    lengths.add(4)

        path = [b]
        seen = {b, source}

        def search(current: int) -> None:
            for neighbor in intersections[current]["neighbors"]:
                if neighbor < 0 or neighbor == source:
                    continue
                if neighbor == b:
                    if 4 < len(path) <= maximum:
                        lengths.add(len(path))
                    continue
                if len(path) >= maximum or neighbor in seen:
                    continue
                seen.add(neighbor)
                path.append(neighbor)
                search(neighbor)
                path.pop()
                seen.remove(neighbor)

        search(b)
    return lengths


def p2_cycle_coverage(maps_dir: Path) -> dict:
    output = {}
    for path in sorted(maps_dir.glob("*_unique_goals.map")):
        topology = build_topology(path)
        lengths = [cycle_lengths(topology, source) for source in range(len(topology))]
        no_four = [index for index, values in enumerate(lengths) if 4 not in values]
        rescued6 = [index for index in no_four if any(value in {5, 6} for value in lengths[index])]
        rescued8 = [index for index in no_four if any(5 <= value <= 8 for value in lengths[index])]
        output[path.stem.removesuffix("_unique_goals")] = {
            "map_file": str(path.relative_to(ROOT)),
            "intersections": len(topology), "without_four_cycle": len(no_four),
            "without_four_cycle_fraction": len(no_four) / max(1, len(topology)),
            "rescued_by_L6": len(rescued6), "rescued_by_L8": len(rescued8),
            "still_uncovered_L8": len(no_four) - len(rescued8),
        }
    return output


def p3_telemetry_audit(records_dir: Path) -> dict:
    sample_metric_dirs = []
    parent = records_dir.parent / "metrics"
    for path in sorted(parent.glob("*")):
        if path.is_dir():
            sample_metric_dirs.append(path)
        if len(sample_metric_dirs) == 3:
            break
    files = sorted({child.name for directory in sample_metric_dirs for child in directory.glob("*.csv")})
    return {
        "available": False,
        "requested_metric": "admission into a direction whose downstream availability is non-positive",
        "reason": (
            "Legacy records expose aggregate gate_signal exchanges but do not log each admission "
            "decision together with its route direction and the one-cycle-stale downstream availability. "
            "The requested count cannot be reconstructed without inventing a proxy."
        ),
        "audited_metric_files": files,
        "required_instrumentation": (
            "At the existing admission decision point, shadow-log source, downstream, direction, "
            "downstream availability, decision, and whether the robot actually entered."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--maps", type=Path, default=DEFAULT_MAPS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {
        "schema_version": 1,
        "p1_completion_tail": p1_completion_tail(args.records.resolve()),
        "p2_cycle_coverage": p2_cycle_coverage(args.maps.resolve()),
        "p3_saturated_admission": p3_telemetry_audit(args.records.resolve()),
    }
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(args.output)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
