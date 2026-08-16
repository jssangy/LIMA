#!/usr/bin/env python3
"""Execute PIBT or LaCAM plans under the common stochastic-delay trace.

Whenever a delayed or safety-cancelled move makes the observed configuration
diverge from the joint plan, the adapter replans from the observed positions to
the remaining goals before the next execution step.  Completed agents
disappear.  The adapter never uses a wall-clock cutoff.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


MASK64 = (1 << 64) - 1
TRACE_SCALE = 1 << 53


def splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def command_delayed(seed: int, agent_id: int, timestep: int, probability: float) -> bool:
    if probability <= 0.0:
        return False
    counter = (seed ^ 0xA0761D6478BD642F) & MASK64
    counter ^= ((agent_id + 1) * 0xD2B74407B1CE6E93) & MASK64
    counter ^= ((timestep + 1) * 0xCA5A826395121157) & MASK64
    return (splitmix64(counter) >> 11) < int(probability * TRACE_SCALE)


def parse_map_size(path: Path) -> tuple[int, int]:
    width = height = None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("height "):
                height = int(line.split()[1])
            elif line.startswith("width "):
                width = int(line.split()[1])
            elif line.strip() == "map":
                break
    if width is None or height is None:
        raise ValueError(f"invalid MovingAI map header: {path}")
    return width, height


def parse_scenario(path: Path, agents: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    rows = [line.split() for line in path.read_text(encoding="utf-8").splitlines()[1:] if line.strip()]
    if len(rows) < agents:
        raise ValueError(f"scenario has {len(rows)} rows, expected {agents}")
    starts = [(int(row[4]), int(row[5])) for row in rows[:agents]]
    goals = [(int(row[6]), int(row[7])) for row in rows[:agents]]
    return starts, goals


def parse_fields(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line == "solution=":
            break
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key.strip()] = value.strip()
    return fields


def parse_plan(path: Path, agents: int) -> list[list[tuple[int, int]]]:
    configurations: list[list[tuple[int, int]]] = []
    in_solution = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line == "solution=":
            in_solution = True
            continue
        if not in_solution or not re.match(r"^\d+:", line):
            continue
        coords = [(int(x), int(y)) for x, y in re.findall(r"\((-?\d+),(-?\d+)\)", line)]
        if len(coords) != agents:
            raise ValueError(f"plan row has {len(coords)} agents, expected {agents}")
        configurations.append(coords)
    if not configurations:
        raise ValueError(f"planner returned no configurations: {path}")
    return configurations


def validate_plan(
    plan: list[list[tuple[int, int]]],
    starts: list[tuple[int, int]],
    goals: list[tuple[int, int]],
) -> None:
    if plan[0] != starts:
        raise ValueError("planner configuration zero does not match observed starts")
    disappeared: set[int] = set()
    for previous, current in zip(plan, plan[1:]):
        active = [index for index in range(len(current)) if index not in disappeared]
        if len({current[index] for index in active}) != len(active):
            raise ValueError("planner returned a vertex conflict")
        for index in active:
            source, target = previous[index], current[index]
            if abs(source[0] - target[0]) + abs(source[1] - target[1]) > 1:
                raise ValueError(f"non-adjacent planner move for agent {index}")
        transitions = {
            (previous[index], current[index])
            for index in active if previous[index] != current[index]
        }
        if any((target, source) in transitions for source, target in transitions):
            raise ValueError("planner returned an edge conflict")
        disappeared.update(index for index in active if current[index] == goals[index])


def write_lacam_scenario(
    path: Path,
    map_path: Path,
    starts: list[tuple[int, int]],
    goals: list[tuple[int, int]],
) -> None:
    width, height = parse_map_size(map_path)
    lines = ["version 1"]
    for start, goal in zip(starts, goals):
        lines.append(
            f"0\t{map_path.name}\t{width}\t{height}\t{start[0]}\t{start[1]}\t"
            f"{goal[0]}\t{goal[1]}\t0"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_pibt_map(map_path: Path, repo: Path) -> str:
    digest = hashlib.sha256(map_path.read_bytes()).hexdigest()
    name = f"lima_{digest[:16]}.map"
    cache = repo / "map" / name
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and hashlib.sha256(cache.read_bytes()).hexdigest() != digest:
        raise ValueError(f"PIBT map cache hash mismatch: {cache}")
    if not cache.exists():
        temporary = cache.with_suffix(".tmp")
        shutil.copyfile(map_path, temporary)
        os.replace(temporary, cache)
    return name


def write_pibt_instance(
    path: Path,
    map_name: str,
    starts: list[tuple[int, int]],
    goals: list[tuple[int, int]],
    seed: int,
    remaining_steps: int,
) -> None:
    lines = [
        f"map_file={map_name}",
        f"agents={len(starts)}",
        f"seed={seed}",
        "random_problem=0",
        f"max_timestep={max(1, remaining_steps)}",
        "max_comp_time=2147483647",
    ]
    lines.extend(
        f"{start[0]},{start[1]},{goal[0]},{goal[1]}"
        for start, goal in zip(starts, goals)
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def safe_execute(
    current: list[tuple[int, int]],
    proposed: list[tuple[int, int]],
    delayed: set[int],
) -> tuple[list[tuple[int, int]], int]:
    moving = [source != target and index not in delayed for index, (source, target) in enumerate(zip(current, proposed))]
    cancelled_by_safety: set[int] = set()
    occupancy = {cell: index for index, cell in enumerate(current)}

    changed = True
    while changed:
        changed = False
        targets = [proposed[i] if moving[i] else current[i] for i in range(len(current))]
        for index, target in enumerate(targets):
            if not moving[index]:
                continue
            occupant = occupancy.get(target)
            if occupant is not None and occupant != index and not moving[occupant]:
                moving[index] = False
                cancelled_by_safety.add(index)
                changed = True

        targets = [proposed[i] if moving[i] else current[i] for i in range(len(current))]
        by_target: dict[tuple[int, int], list[int]] = {}
        for index, target in enumerate(targets):
            by_target.setdefault(target, []).append(index)
        for indices in by_target.values():
            if len(indices) < 2:
                continue
            for index in indices:
                if moving[index]:
                    moving[index] = False
                    cancelled_by_safety.add(index)
                    changed = True

        for i in range(len(current)):
            if not moving[i]:
                continue
            for j in range(i + 1, len(current)):
                if moving[j] and proposed[i] == current[j] and proposed[j] == current[i]:
                    moving[i] = moving[j] = False
                    cancelled_by_safety.update((i, j))
                    changed = True

    actual = [proposed[i] if moving[i] else current[i] for i in range(len(current))]
    if len(set(actual)) != len(actual):
        raise ValueError("safe executor produced a vertex conflict")
    transitions = {(source, target) for source, target in zip(current, actual) if source != target}
    if any((target, source) in transitions for source, target in transitions):
        raise ValueError("safe executor produced an edge conflict")
    return actual, len(cancelled_by_safety)


class Planner:
    def __init__(self, args: argparse.Namespace, temporary: Path, map_path: Path):
        self.args = args
        self.temporary = temporary
        self.map_path = map_path
        self.calls = 0
        self.total_comp_time_ms = 0
        self.total_search_iterations = 0
        self.pibt_map_name = (
            ensure_pibt_map(map_path, args.pibt_repo)
            if args.algorithm == "pibt" else None
        )

    def plan(
        self,
        starts: list[tuple[int, int]],
        goals: list[tuple[int, int]],
        remaining_steps: int,
    ) -> tuple[list[list[tuple[int, int]]] | None, str]:
        self.calls += 1
        instance = self.temporary / f"instance_{self.calls:05d}.txt"
        output = self.temporary / f"solution_{self.calls:05d}.txt"
        if self.args.algorithm == "lacam":
            write_lacam_scenario(instance, self.map_path, starts, goals)
            command = [
                str(self.args.lacam_binary), "--map", str(self.map_path),
                "--scen", str(instance), "--num", str(len(starts)),
                "--seed", str(self.args.seed), "--time_limit_sec", "0",
                "--max_iterations", str(self.args.lacam_max_iterations),
                "--disappear-at-goal", "--output", str(output),
            ]
            cwd = self.args.lacam_repo
        else:
            write_pibt_instance(
                instance, self.pibt_map_name, starts, goals,
                self.args.seed, remaining_steps,
            )
            command = [
                str(self.args.pibt_binary), "--instance", str(instance),
                "--solver", "PIBT", "--output", str(output),
                "--disappear-at-goal",
            ]
            cwd = self.args.pibt_repo
        process = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
        if process.returncode != 0 or not output.is_file():
            return None, f"planner_returncode_{process.returncode}"
        fields = parse_fields(output)
        try:
            self.total_comp_time_ms += int(fields.get("comp_time", "0"))
            self.total_search_iterations += int(fields.get("search_iterations", "0"))
        except ValueError:
            return None, "invalid_planner_metrics"
        if fields.get("solved") != "1":
            return None, "search_limit"
        plan = parse_plan(output, len(starts))
        validate_plan(plan, starts, goals)
        return plan, "solved"


def emit(payload: dict[str, object]) -> None:
    print(" ".join(f"{key}={value}" for key, value in payload.items()), flush=True)


def route_suffixes(
    plan: list[list[tuple[int, int]]], plan_ids: list[int], start_index: int,
    active_ids: list[int], goals: list[tuple[int, int]],
) -> dict[int, tuple[tuple[int, int], ...]]:
    """Return future cached-route suffixes, trimmed at first goal arrival."""
    index_by_id = {agent_id: index for index, agent_id in enumerate(plan_ids)}
    suffixes: dict[int, tuple[tuple[int, int], ...]] = {}
    for agent_id, goal in zip(active_ids, goals):
        local_index = index_by_id.get(agent_id)
        if local_index is None:
            suffixes[agent_id] = ()
            continue
        cells: list[tuple[int, int]] = []
        for frame in plan[start_index:]:
            cell = frame[local_index]
            cells.append(cell)
            if cell == goal:
                break
        suffixes[agent_id] = tuple(cells)
    return suffixes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", required=True, choices=("pibt", "lacam"))
    parser.add_argument("--map", required=True, type=Path)
    parser.add_argument("--scen", required=True, type=Path)
    parser.add_argument("--agents", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--max-steps", required=True, type=int)
    parser.add_argument("--delay-prob", required=True, type=float)
    parser.add_argument("--delay-seed", required=True, type=int)
    parser.add_argument("--pibt-repo", type=Path, default=Path.home() / "mapf-baselines/pibt2")
    parser.add_argument("--pibt-binary", type=Path, default=Path("results/revision_final/frozen_artifacts_step_v2/pibt"))
    parser.add_argument("--lacam-repo", type=Path, default=Path.home() / "mapf-baselines/lacam")
    parser.add_argument("--lacam-binary", type=Path, default=Path("results/revision_final/frozen_artifacts_step_v2/lacam"))
    parser.add_argument("--lacam-max-iterations", type=int, default=100000)
    parser.add_argument("--trace-output", type=Path)
    args = parser.parse_args()
    args.map = args.map.resolve()
    args.scen = args.scen.resolve()
    args.pibt_repo = args.pibt_repo.resolve()
    args.pibt_binary = args.pibt_binary.resolve()
    args.lacam_repo = args.lacam_repo.resolve()
    args.lacam_binary = args.lacam_binary.resolve()
    if args.agents < 1 or args.max_steps < 1 or not 0.0 <= args.delay_prob <= 1.0:
        parser.error("invalid agents, max-steps, or delay probability")

    starts, goals = parse_scenario(args.scen, args.agents)
    active_ids = list(range(args.agents))
    positions = list(starts)
    active_goals = list(goals)
    completion_steps = [0] * args.agents
    delayed_moves = 0
    interventions = 0
    deviation_steps = 0
    task_upload_messages = 0
    state_upload_messages = 0
    route_delivery_messages = 0
    route_waypoint_payload = 0
    decision_hash = hashlib.sha256()
    trace: list[dict[str, object]] = []
    status = "step_limit"

    with tempfile.TemporaryDirectory(prefix="lima_stochastic_replan_") as directory:
        planner = Planner(args, Path(directory), args.map)
        plan: list[list[tuple[int, int]]] | None = None
        plan_ids: list[int] = []
        plan_index = 0
        need_replan = True
        steps = 0
        while steps < args.max_steps and active_ids:
            if need_replan:
                initial_plan = plan is None
                if initial_plan:
                    task_upload_messages += len(active_ids)
                else:
                    # A solver call necessarily receives the current state, even
                    # if it later returns an unchanged route or fails.
                    state_upload_messages += len(active_ids)
                old_suffixes = (
                    {}
                    if initial_plan else route_suffixes(
                        plan, plan_ids, min(plan_index + 2, len(plan)),
                        active_ids, active_goals,
                    )
                )
                plan, plan_status = planner.plan(
                    positions, active_goals, args.max_steps - steps
                )
                if plan is None:
                    status = plan_status
                    break
                plan_ids = list(active_ids)
                plan_index = 0
                new_suffixes = route_suffixes(
                    plan, plan_ids, 1, active_ids, active_goals
                )
                changed_ids = [
                    agent_id for agent_id in active_ids
                    if initial_plan or new_suffixes[agent_id] != old_suffixes.get(agent_id)
                ]
                route_delivery_messages += len(changed_ids)
                route_waypoint_payload += sum(
                    len(new_suffixes[agent_id]) for agent_id in changed_ids
                )
                need_replan = False

            if plan_index + 1 >= len(plan):
                need_replan = True
                continue
            plan_lookup = {
                agent_id: plan[plan_index + 1][index]
                for index, agent_id in enumerate(plan_ids)
            }
            proposed = [plan_lookup[agent_id] for agent_id in active_ids]
            delayed = {
                local_index
                for local_index, (agent_id, source, target) in enumerate(
                    zip(active_ids, positions, proposed)
                )
                if source != target
                and command_delayed(
                    args.delay_seed, agent_id, steps, args.delay_prob
                )
            }
            delayed_moves += len(delayed)
            actual, safety_cancelled = safe_execute(positions, proposed, delayed)
            interventions += safety_cancelled
            diverged = actual != proposed
            deviation_steps += int(diverged)
            steps += 1

            decision_hash.update(steps.to_bytes(8, "little"))
            for agent_id, cell in zip(active_ids, actual):
                decision_hash.update(agent_id.to_bytes(8, "little"))
                decision_hash.update(cell[0].to_bytes(4, "little", signed=True))
                decision_hash.update(cell[1].to_bytes(4, "little", signed=True))

            finished = [index for index, (cell, goal) in enumerate(zip(actual, active_goals)) if cell == goal]
            for index in finished:
                completion_steps[active_ids[index]] = steps
            if args.trace_output is not None:
                trace.append({
                    "step": steps,
                    "active": len(active_ids),
                    "delayed_moves": len(delayed),
                    "safe_executor_interventions": safety_cancelled,
                    "completed": len(finished),
                    "replan_next": bool(diverged),
                })

            keep = [index for index in range(len(active_ids)) if index not in set(finished)]
            active_ids = [active_ids[index] for index in keep]
            positions = [actual[index] for index in keep]
            active_goals = [active_goals[index] for index in keep]
            if not active_ids:
                status = "completed"
                break
            if diverged:
                need_replan = True
            else:
                plan_index += 1

        completed = sum(step > 0 for step in completion_steps)
        solved = completed == args.agents
        if args.trace_output is not None:
            args.trace_output.parent.mkdir(parents=True, exist_ok=True)
            args.trace_output.write_text(
                __import__("json").dumps(trace, indent=2) + "\n", encoding="utf-8"
            )
        emit({
            "status": "completed" if solved else status,
            "solved": int(solved),
            "completed": f"{completed}/{args.agents}",
            "steps": steps,
            "makespan": steps if solved else args.max_steps,
            "soc": sum(completion_steps),
            "planning_calls": planner.calls,
            "replans": max(0, planner.calls - 1),
            "planning_comp_time_ms": planner.total_comp_time_ms,
            "search_iterations": planner.total_search_iterations,
            "delayed_moves": delayed_moves,
            "deviation_steps": deviation_steps,
            "safe_executor_interventions": interventions,
            "communication_events": (
                task_upload_messages + state_upload_messages
                + route_delivery_messages
            ),
            "communication_task_uploads": task_upload_messages,
            "communication_state_uploads": state_upload_messages,
            "communication_route_deliveries": route_delivery_messages,
            "communication_route_waypoints": route_waypoint_payload,
            "communication_cached_route_step_events": 0,
            "communication_scope": "global",
            "recovery_latency_steps": 1 if deviation_steps else 0,
            "vertex_conflicts": 0,
            "edge_conflicts": 0,
            "decision_hash": decision_hash.hexdigest(),
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
