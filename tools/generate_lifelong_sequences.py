#!/usr/bin/env python3
"""Generate fixed, disjoint, cyclic per-agent goal sequences for lifelong runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))
DENSITIES = (10, 30, 50)


@dataclass(frozen=True)
class Spec:
    map_file: str
    scenario_template: str
    tiles: int
    seed_salt: int


SPECS = {
    "warehouse_10_20": Spec(
        "data/maps/warehouse_10_20_paper.map",
        "data/scenarios/warehouse-10-20-paper/warehouse-10-20-paper_s{s}.scen",
        2649, 0x1020),
    "warehouse_20_40": Spec(
        "data/maps/warehouse_20_40_paper.map",
        "data/scenarios/warehouse-20-40-paper/warehouse-20-40-paper_s{s}.scen",
        10499, 0x2040),
    "cross_3030": Spec(
        "data/maps/cross_3030_paper.map",
        "data/scenarios/cross-30-30-paper/cross-30-30-paper_s{s}.scen",
        10200, 0x3030),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8", newline="\n"
    ) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8"
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def parse_int_list(text: str, allowed: set[int]) -> list[int]:
    values: set[int] = set()
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            lower, upper = (int(value) for value in item.split("-", 1))
            values.update(range(lower, upper + 1))
        else:
            values.add(int(item))
    if not values or not values.issubset(allowed):
        raise argparse.ArgumentTypeError(f"values must be a nonempty subset of {sorted(allowed)}")
    return sorted(values)


def load_map(path: Path) -> tuple[int, int, list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    height, width = int(lines[1].split()[1]), int(lines[2].split()[1])
    rows = lines[4:4 + height]
    if lines[3].strip() != "map" or len(rows) != height or any(len(row) != width for row in rows):
        raise ValueError(f"invalid MovingAI map: {path}")
    return width, height, rows


def load_starts(path: Path, count: int) -> list[tuple[int, int]]:
    starts = []
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        fields = line.split()
        if len(fields) >= 8:
            starts.append((int(fields[4]), int(fields[5])))
        if len(starts) == count:
            break
    if len(starts) != count:
        raise ValueError(f"{path} contains only {len(starts)} starts; need {count}")
    return starts


def component_labels(width: int, height: int, rows: list[str]) -> dict[tuple[int, int], int]:
    traversable = {
        (x, y) for y, row in enumerate(rows) for x, value in enumerate(row)
        if value in ".SEG"
    }
    labels: dict[tuple[int, int], int] = {}
    label = 0
    for start in sorted(traversable):
        if start in labels:
            continue
        labels[start] = label
        queue = deque([start])
        while queue:
            x, y = queue.popleft()
            for dx, dy in DELTAS:
                neighbor = (x + dx, y + dy)
                if neighbor in traversable and neighbor not in labels:
                    labels[neighbor] = label
                    queue.append(neighbor)
        label += 1
    return labels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps", default=",".join(SPECS))
    parser.add_argument("--densities", default=",".join(map(str, DENSITIES)))
    parser.add_argument("--scenarios", default="0-9")
    parser.add_argument("--output-root", default="results/revision_final/lifelong_inputs_v1")
    args = parser.parse_args()
    maps = [item.strip() for item in args.maps.split(",") if item.strip()]
    if not maps or not set(maps).issubset(SPECS):
        parser.error("unknown or empty map selection")
    densities = parse_int_list(args.densities, set(DENSITIES))
    scenarios = parse_int_list(args.scenarios, set(range(10)))
    output = (ROOT / args.output_root).resolve()
    script = Path(__file__).resolve()
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generator": str(script.relative_to(ROOT)), "generator_sha256": sha256(script),
        "semantics": {
            "goal_pool": "interior traversable non-sink cells",
            "assignment": "disjoint per-agent subsets",
            "sequence": "fixed cyclic order; no consecutive duplicate",
            "activation": "next goal in the completion timestep; movement starts next timestep",
        },
        "maps": {},
    }
    generated = 0
    for map_name in maps:
        spec = SPECS[map_name]
        map_path = ROOT / spec.map_file
        width, height, rows = load_map(map_path)
        pool = [
            (x, y) for y in range(1, height - 1) for x in range(1, width - 1)
            if rows[y][x] in ".EG"
        ]
        labels = component_labels(width, height, rows)
        map_entry = {
            "map_file": spec.map_file, "map_sha256": sha256(map_path),
            "tiles": spec.tiles, "goal_pool_size": len(pool), "densities": {},
        }
        manifest["maps"][map_name] = map_entry
        for density in densities:
            agents = density * spec.tiles // 100
            density_entry = {"agents": agents, "scenarios": {}}
            map_entry["densities"][str(density)] = density_entry
            for scenario in scenarios:
                source = ROOT / spec.scenario_template.format(s=scenario)
                starts = load_starts(source, agents)
                if len(set(starts)) != agents:
                    raise AssertionError("source scenario has duplicate starts")
                seed = (spec.seed_salt << 32) ^ (density << 16) ^ scenario
                buckets: list[list[tuple[int, int]]] = [[] for _ in range(agents)]
                reachable_pool_size = 0
                start_components = sorted({labels[start] for start in starts})
                for component in start_components:
                    agent_indexes = [
                        index for index, start in enumerate(starts) if labels[start] == component
                    ]
                    component_pool = [goal for goal in pool if labels[goal] == component]
                    if len(component_pool) < 2 * len(agent_indexes):
                        raise ValueError(
                            f"{map_name} d{density} component {component}: "
                            "reachable goal pool cannot give two cells per agent")
                    random.Random(seed ^ component).shuffle(component_pool)
                    reachable_pool_size += len(component_pool)
                    for local_index, agent_index in enumerate(agent_indexes):
                        buckets[agent_index] = component_pool[local_index::len(agent_indexes)]
                for index, bucket in enumerate(buckets):
                    if len(bucket) < 2:
                        raise AssertionError("lifelong sequence has fewer than two unique goals")
                    if bucket[0] == starts[index]:
                        bucket.append(bucket.pop(0))
                    if bucket[0] == starts[index]:
                        raise AssertionError("initial lifelong goal equals start")
                    if any(labels[start] != labels[goal] for start, goal in zip([starts[index]] * len(bucket), bucket)):
                        raise AssertionError("lifelong sequence contains an unreachable goal")
                all_goals = [goal for bucket in buckets for goal in bucket]
                if len(set(all_goals)) != reachable_pool_size:
                    raise AssertionError("per-agent lifelong goal subsets overlap or omit cells")

                tag = f"{map_name}_d{density:02d}_a{agents}_s{scenario}"
                scenario_path = output / "scenarios" / map_name / f"{tag}.scen"
                sequence_path = output / "sequences" / map_name / f"{tag}.txt"
                scenario_rows = ["version 1"]
                for start, bucket in zip(starts, buckets):
                    scenario_rows.append(
                        f"0\t{Path(spec.map_file).name}\t{width}\t{height}\t"
                        f"{start[0]}\t{start[1]}\t{bucket[0][0]}\t{bucket[0][1]}\t0")
                sequence_rows = [
                    " ".join(f"{x} {y}" for x, y in bucket) for bucket in buckets
                ]
                atomic_text(scenario_path, "\n".join(scenario_rows) + "\n")
                atomic_text(sequence_path, "\n".join(sequence_rows) + "\n")
                certificate_path = output / "certificates" / map_name / f"{tag}.json"
                certificate = {
                    "map": map_name, "density_percent": density, "agents": agents,
                    "scenario": scenario, "seed": seed,
                    "source_start_scenario": str(source.relative_to(ROOT)),
                    "source_start_scenario_sha256": sha256(source),
                    "scenario_file": str(scenario_path.relative_to(ROOT)),
                    "scenario_sha256": sha256(scenario_path),
                    "sequence_file": str(sequence_path.relative_to(ROOT)),
                    "sequence_sha256": sha256(sequence_path),
                    "validation": {
                        "unique_starts": True, "disjoint_goal_subsets": True,
                        "reachable_goals": len(all_goals),
                        "reachable_goal_pool_size": reachable_pool_size,
                        "minimum_goals_per_agent": min(map(len, buckets)),
                        "maximum_goals_per_agent": max(map(len, buckets)),
                        "initial_start_goal_collisions": 0,
                    },
                }
                atomic_json(certificate_path, certificate)
                density_entry["scenarios"][str(scenario)] = {
                    "scenario_file": str(scenario_path.relative_to(ROOT)),
                    "scenario_sha256": sha256(scenario_path),
                    "sequence_file": str(sequence_path.relative_to(ROOT)),
                    "sequence_sha256": sha256(sequence_path),
                    "certificate_file": str(certificate_path.relative_to(ROOT)),
                    "certificate_sha256": sha256(certificate_path),
                }
                generated += 1
                print(
                    f"[{generated:3d}] {tag}: goals/agent="
                    f"{min(map(len, buckets))}-{max(map(len, buckets))}", flush=True)
    atomic_json(output / "MANIFEST.json", manifest)
    print(f"generated {generated} fixed lifelong inputs under {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
