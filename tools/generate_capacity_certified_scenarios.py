#!/usr/bin/env python3
"""Generate operational-capacity-certified one-shot start placements.

Starts are restricted to the managed-cell union.  For every managed
intersection, the selected starts satisfy the frozen sum-minus-max occupancy
bound.  Goals are copied from the matching submitted scenario so the physical
sink distribution remains fixed while only the initial placement changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, vstack


ROOT = Path(__file__).resolve().parent.parent
DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))
DIAGONALS = ((-1, -1), (1, -1), (1, 1), (-1, 1))
FREE = frozenset(".SEG")
GOAL = frozenset("SG")


@dataclass(frozen=True)
class Spec:
    map_file: str
    scenario_template: str
    tiles: int
    targets: tuple[int | str, ...]
    seed_salt: int


SPECS = {
    "warehouse_10_20": Spec(
        "data/maps/warehouse_10_20_paper.map",
        "data/scenarios/warehouse-10-20-paper/warehouse-10-20-paper_s{s}.scen",
        2649, (60, "boundary"), 0x1020),
    "warehouse_20_40": Spec(
        "data/maps/warehouse_20_40_paper.map",
        "data/scenarios/warehouse-20-40-paper/warehouse-20-40-paper_s{s}.scen",
        10499, (50, "boundary"), 0x2040),
    "cross_3030": Spec(
        "data/maps/cross_3030_paper.map",
        "data/scenarios/cross-30-30-paper/cross-30-30-paper_s{s}.scen",
        10200, (60, 65, 70, "boundary"), 0x3030),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8"
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8", newline="\n"
    ) as stream:
        stream.write(text)
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


class Grid:
    def __init__(self, path: Path):
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) < 4 or lines[3].strip() != "map":
            raise ValueError(f"not a MovingAI map: {path}")
        self.height = int(lines[1].split()[1])
        self.width = int(lines[2].split()[1])
        self.rows = lines[4:4 + self.height]
        if len(self.rows) != self.height or any(len(row) != self.width for row in self.rows):
            raise ValueError(f"map dimensions do not match header: {path}")
        self.traversable = {
            self.cell(x, y)
            for y, row in enumerate(self.rows)
            for x, value in enumerate(row)
            if value in FREE
        }
        self.goals = {
            self.cell(x, y)
            for y, row in enumerate(self.rows)
            for x, value in enumerate(row)
            if value in GOAL
        }

    def cell(self, x: int, y: int) -> int:
        return y * self.width + x

    def coord(self, cell: int) -> tuple[int, int]:
        return cell % self.width, cell // self.width

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_traversable(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and self.cell(x, y) in self.traversable


def detect_center(grid: Grid, x: int, y: int) -> bool:
    if not grid.is_traversable(x, y):
        return False
    if sum(grid.is_traversable(x + dx, y + dy) for dx, dy in DELTAS) < 3:
        return False
    return all(not grid.is_traversable(x + dx, y + dy) for dx, dy in DIAGONALS)


def trace_arm(grid: Grid, center: tuple[int, int], direction: int) -> list[int]:
    dx, dy = DELTAS[direction]
    x, y = center[0] + dx, center[1] + dy
    cells: list[int] = []
    while grid.is_traversable(x, y):
        cell = grid.cell(x, y)
        if cell in grid.goals:
            break
        corridor = (
            (not grid.is_traversable(x - 1, y) and not grid.is_traversable(x + 1, y))
            if dy else
            (not grid.is_traversable(x, y - 1) and not grid.is_traversable(x, y + 1))
        )
        if not corridor:
            break
        cells.append(cell)
        x, y = x + dx, y + dy
    return cells


def build_topology(grid: Grid) -> list[dict]:
    intersections: list[dict] = []
    for cell in sorted(grid.traversable):
        center = grid.coord(cell)
        if not detect_center(grid, *center):
            continue
        arms = [trace_arm(grid, center, direction) for direction in range(4)]
        if sum(bool(arm) for arm in arms) < 3:
            continue
        intersections.append({"center": cell, "arms": arms})
    return intersections


def component_labels(grid: Grid) -> dict[int, int]:
    labels: dict[int, int] = {}
    label = 0
    for start in sorted(grid.traversable):
        if start in labels:
            continue
        labels[start] = label
        queue = deque([start])
        while queue:
            current = queue.popleft()
            x, y = grid.coord(current)
            for dx, dy in DELTAS:
                neighbor = grid.cell(x + dx, y + dy) if grid.in_bounds(x + dx, y + dy) else -1
                if neighbor in grid.traversable and neighbor not in labels:
                    labels[neighbor] = label
                    queue.append(neighbor)
        label += 1
    return labels


def parse_goals(path: Path) -> list[tuple[int, int]]:
    goals: list[tuple[int, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        fields = line.split()
        if len(fields) >= 8:
            goals.append((int(fields[6]), int(fields[7])))
    return goals


def make_constraints(
    intersections: list[dict], managed_cells: list[int]
) -> tuple[csr_matrix, np.ndarray]:
    index = {cell: column for column, cell in enumerate(managed_cells)}
    rows: list[int] = []
    columns: list[int] = []
    for row, intersection in enumerate(intersections):
        cells = [intersection["center"]]
        for arm in intersection["arms"]:
            cells.extend(arm)
        for cell in set(cells):
            rows.append(row)
            columns.append(index[cell])
    matrix = csr_matrix(
        (np.ones(len(rows)), (np.array(rows), np.array(columns))),
        shape=(len(intersections), len(managed_cells)), dtype=float)
    capacities = np.array([
        sum(len(arm) for arm in intersection["arms"])
        - max(len(arm) for arm in intersection["arms"])
        for intersection in intersections
    ], dtype=float)
    return matrix, capacities


def solve_maximum(matrix: csr_matrix, capacities: np.ndarray) -> tuple[int, dict]:
    variables = matrix.shape[1]
    result = milp(
        c=-np.ones(variables), integrality=np.ones(variables), bounds=Bounds(0, 1),
        constraints=LinearConstraint(matrix, -np.inf, capacities),
        options={"time_limit": 300, "mip_rel_gap": 0.0})
    if not result.success or result.fun is None:
        raise RuntimeError(f"maximum placement ILP failed: {result.message}")
    maximum = int(round(-float(result.fun)))
    return maximum, {
        "status": int(result.status), "message": result.message,
        "objective": maximum,
        "mip_gap": float(getattr(result, "mip_gap", 0.0)),
        "mip_node_count": int(getattr(result, "mip_node_count", 0)),
        "mip_dual_bound": float(-getattr(result, "mip_dual_bound", -maximum)),
    }


def solve_target(
    matrix: csr_matrix, capacities: np.ndarray, target: int, seed: int
) -> tuple[np.ndarray, dict]:
    variables = matrix.shape[1]
    rng = random.Random(seed)
    objective = np.array([rng.random() for _ in range(variables)], dtype=float)
    augmented = vstack([matrix, csr_matrix(np.ones((1, variables)))], format="csr")
    lower = np.concatenate([np.full(matrix.shape[0], -np.inf), [target]])
    upper = np.concatenate([capacities, [target]])
    result = milp(
        c=objective, integrality=np.ones(variables), bounds=Bounds(0, 1),
        constraints=LinearConstraint(augmented, lower, upper),
        options={"time_limit": 300, "mip_rel_gap": 0.0})
    if not result.success or result.x is None:
        raise RuntimeError(f"target placement ILP failed for N={target}: {result.message}")
    selected = np.flatnonzero(result.x > 0.5)
    if len(selected) != target:
        raise AssertionError(f"ILP returned {len(selected)} starts; expected {target}")
    return selected, {
        "status": int(result.status), "message": result.message,
        "random_objective": float(result.fun),
        "mip_gap": float(getattr(result, "mip_gap", 0.0)),
        "mip_node_count": int(getattr(result, "mip_node_count", 0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps", default=",".join(SPECS))
    parser.add_argument("--scenarios", default="0-9")
    parser.add_argument(
        "--output-root", default="results/revision_final/certified_inputs_v1")
    args = parser.parse_args()
    maps = [item.strip() for item in args.maps.split(",") if item.strip()]
    if not maps or not set(maps).issubset(SPECS):
        parser.error("unknown or empty map selection")
    scenarios = parse_int_list(args.scenarios, set(range(10)))
    output = (ROOT / args.output_root).resolve()
    script = Path(__file__).resolve()
    manifest: dict = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generator": str(script.relative_to(ROOT)),
        "generator_sha256": sha256(script),
        "capacity_formula": "sum(arm capacities) - longest arm",
        "start_domain": "managed intersection centers and arms only",
        "maps": {},
    }

    total_files = 0
    for map_name in maps:
        spec = SPECS[map_name]
        map_path = ROOT / spec.map_file
        grid = Grid(map_path)
        topology = build_topology(grid)
        managed_set = {
            cell for intersection in topology
            for cell in [intersection["center"],
                         *(cell for arm in intersection["arms"] for cell in arm)]
        }
        managed_cells = sorted(managed_set)
        if len(managed_cells) != spec.tiles:
            raise AssertionError(
                f"{map_name}: managed cells {len(managed_cells)} != Table 2 tiles {spec.tiles}")
        matrix, capacities = make_constraints(topology, managed_cells)
        maximum, maximum_certificate = solve_maximum(matrix, capacities)
        labels = component_labels(grid)
        map_entry = {
            "map_file": spec.map_file, "map_sha256": sha256(map_path),
            "tiles": spec.tiles, "managed_cells": len(managed_cells),
            "intersections": len(topology), "maximum_agents": maximum,
            "boundary_percent": 100.0 * maximum / spec.tiles,
            "maximum_certificate": maximum_certificate,
            "targets": {},
        }
        manifest["maps"][map_name] = map_entry
        for target_name in spec.targets:
            target = maximum if target_name == "boundary" else int(target_name) * spec.tiles // 100
            if target > maximum:
                raise AssertionError(f"{map_name} target {target} exceeds ILP maximum {maximum}")
            label = "boundary" if target_name == "boundary" else f"d{int(target_name):02d}"
            target_entry = {"agents": target, "scenarios": {}}
            map_entry["targets"][label] = target_entry
            for scenario in scenarios:
                source_scenario = ROOT / spec.scenario_template.format(s=scenario)
                goals = parse_goals(source_scenario)
                if len(goals) < target:
                    raise ValueError(f"{source_scenario} has {len(goals)} tasks; need {target}")
                seed = (spec.seed_salt << 32) ^ (target << 8) ^ scenario
                selected_columns, solve_certificate = solve_target(
                    matrix, capacities, target, seed)
                starts = [managed_cells[int(column)] for column in selected_columns]
                permutation = list(range(len(starts)))
                random.Random(seed ^ 0x9E3779B97F4A7C15).shuffle(permutation)
                starts = [starts[index] for index in permutation]
                occupancies = np.asarray(matrix[:, selected_columns].sum(axis=1)).ravel().astype(int)
                if np.any(occupancies > capacities):
                    raise AssertionError("generated placement violates an intersection capacity")
                rows = ["version 1"]
                for start, goal in zip(starts, goals[:target]):
                    sx, sy = grid.coord(start)
                    gx, gy = goal
                    goal_cell = grid.cell(gx, gy)
                    if start == goal_cell or labels[start] != labels[goal_cell]:
                        raise AssertionError("generated start-goal pair is invalid or unreachable")
                    rows.append(
                        f"0\t{Path(spec.map_file).name}\t{grid.width}\t{grid.height}\t"
                        f"{sx}\t{sy}\t{gx}\t{gy}\t0")
                scenario_dir = output / "scenarios" / map_name
                scenario_path = scenario_dir / f"{map_name}_cert_{label}_s{scenario}.scen"
                atomic_text(scenario_path, "\n".join(rows) + "\n")
                certificate_path = output / "certificates" / map_name / f"{label}_s{scenario}.json"
                certificate = {
                    "map": map_name, "map_file": spec.map_file,
                    "map_sha256": sha256(map_path), "target": label,
                    "agents": target, "scenario": scenario, "seed": seed,
                    "capacity_formula": "sum-minus-max", "maximum_agents": maximum,
                    "source_goal_scenario": str(source_scenario.relative_to(ROOT)),
                    "source_goal_scenario_sha256": sha256(source_scenario),
                    "scenario_file": str(scenario_path.relative_to(ROOT)),
                    "scenario_sha256": sha256(scenario_path),
                    "solver": solve_certificate,
                    "validation": {
                        "unique_starts": len(set(starts)) == target,
                        "managed_starts": all(start in managed_set for start in starts),
                        "reachable_pairs": target,
                        "capacity_violations": int(np.sum(occupancies > capacities)),
                        "maximum_occupancy": int(occupancies.max(initial=0)),
                        "minimum_slack": int(np.min(capacities.astype(int) - occupancies)),
                    },
                    "intersection_constraints": [
                        {"intersection": index, "occupancy": int(occupancy),
                         "capacity": int(capacity), "slack": int(capacity - occupancy)}
                        for index, (occupancy, capacity) in enumerate(zip(occupancies, capacities))
                    ],
                }
                atomic_json(certificate_path, certificate)
                target_entry["scenarios"][str(scenario)] = {
                    "scenario_file": str(scenario_path.relative_to(ROOT)),
                    "scenario_sha256": sha256(scenario_path),
                    "certificate_file": str(certificate_path.relative_to(ROOT)),
                    "certificate_sha256": sha256(certificate_path),
                }
                total_files += 1
                print(
                    f"[{total_files:3d}] {map_name} {label} s{scenario}: "
                    f"N={target}/{maximum} min_slack={certificate['validation']['minimum_slack']}",
                    flush=True)
    atomic_json(output / "MANIFEST.json", manifest)
    print(f"generated {total_files} certified scenarios under {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
