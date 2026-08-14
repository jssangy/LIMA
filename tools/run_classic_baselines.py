#!/usr/bin/env python3
"""Fair LaCAM-vs-PIBT diagnostic on classic unique-goal MAPF instances.

The submitted LIMA scenarios deliberately reuse workstation sinks and use
disappear-at-target semantics. Classic MAPF solvers retain agents at goals and
therefore cannot solve those instances when goals repeat. This runner derives a
separate, explicitly labelled diagnostic: starts are kept exactly, goals are a
cyclic permutation of those unique starts, and every ``S`` cell is exposed as a
normal traversable cell. Both solvers receive the same wall-clock budget.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parent.parent
PIBT_MAP_DIR = Path.home() / "mapf-baselines/pibt2/map"


@dataclass(frozen=True)
class Instance:
    map_file: str
    scenario_template: str
    tiles: int


INSTANCES = {
    "warehouse_10_20": Instance(
        "data/maps/warehouse_10_20_paper.map",
        "data/scenarios/warehouse-10-20-paper/warehouse-10-20-paper_s{s}.scen",
        2649,
    ),
    "warehouse_20_40": Instance(
        "data/maps/warehouse_20_40_paper.map",
        "data/scenarios/warehouse-20-40-paper/warehouse-20-40-paper_s{s}.scen",
        10499,
    ),
    "cross_3030": Instance(
        "data/maps/cross_3030_paper.map",
        "data/scenarios/cross-30-30-paper/cross-30-30-paper_s{s}.scen",
        10200,
    ),
}


def parse_int_list(text: str, allowed: set[int]) -> list[int]:
    values: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(v) for v in part.split("-", 1))
            values.update(range(lo, hi + 1))
        else:
            values.add(int(part))
    if not values.issubset(allowed):
        raise argparse.ArgumentTypeError(f"values must be a subset of {sorted(allowed)}")
    return sorted(values)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_movingai(path: Path, count: int):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        fields = line.split()
        if len(fields) < 8:
            continue
        rows.append(fields)
        if len(rows) == count:
            break
    if len(rows) != count:
        raise ValueError(f"{path} contains only {len(rows)} usable tasks; need {count}")
    starts = [(int(row[4]), int(row[5])) for row in rows]
    if len(set(starts)) != len(starts):
        raise ValueError(f"starts are not unique in {path} prefix N={count}")
    shift = max(1, count // 2)
    goals = starts[shift:] + starts[:shift]
    if any(start == goal for start, goal in zip(starts, goals)):
        raise AssertionError("cyclic goal permutation produced a fixed point")
    return rows, starts, goals


def prepare_cell(root: Path, map_name: str, density: int, agents: int, scenario: int):
    instance = INSTANCES[map_name]
    map_source = ROOT / instance.map_file
    map_target = root / "maps" / f"{map_name}.map"
    map_target.parent.mkdir(parents=True, exist_ok=True)
    if not map_target.exists():
        # LaCAM treats only T/@ as obstacles. PIBT expects the classic alphabet;
        # exposing S as '.' gives both exactly the same traversable graph.
        map_target.write_text(map_source.read_text(encoding="utf-8").replace("S", "."), encoding="utf-8")

    source_scen = ROOT / instance.scenario_template.format(s=scenario)
    rows, starts, goals = parse_movingai(source_scen, agents)
    tag = f"{map_name}_d{density:02d}_a{agents}_s{scenario}"
    scen_target = root / "scenarios" / f"{tag}.scen"
    scen_target.parent.mkdir(parents=True, exist_ok=True)
    if not scen_target.exists():
        out = ["version 1"]
        for fields, start, goal in zip(rows, starts, goals):
            fields = list(fields)
            fields[1] = map_target.name
            fields[4], fields[5] = str(start[0]), str(start[1])
            fields[6], fields[7] = str(goal[0]), str(goal[1])
            fields[8] = "0"
            out.append("\t".join(fields))
        scen_target.write_text("\n".join(out) + "\n", encoding="utf-8")

    pibt_target = root / "pibt_instances" / f"{tag}.txt"
    pibt_target.parent.mkdir(parents=True, exist_ok=True)
    if not pibt_target.exists():
        # The upstream PIBT2 build hardcodes _MAPDIR_; a relative path is
        # therefore resolved from its map/ directory even when the instance
        # file itself lives elsewhere.
        pibt_map_file = os.path.relpath(map_target, PIBT_MAP_DIR)
        lines = [
            f"map_file={pibt_map_file}", f"agents={agents}", f"seed={scenario}",
            "random_problem=0", "max_timestep=100000",
        ]
        lines += [f"{sx},{sy},{gx},{gy}" for (sx, sy), (gx, gy) in zip(starts, goals)]
        pibt_target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tag, map_target, scen_target, pibt_target


def parse_result(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    return dict(re.findall(r"^(\w+)=([^\n]+)$", path.read_text(encoding="utf-8", errors="replace"), re.MULTILINE))


def run_measured(command: list[str], timeout: float):
    proc = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, start_new_session=True)
    stopped = threading.Event()
    peak_rss = 0

    def monitor() -> None:
        nonlocal peak_rss
        process = psutil.Process(proc.pid)
        while not stopped.is_set():
            try:
                family = [process, *process.children(recursive=True)]
                peak_rss = max(peak_rss, sum(item.memory_info().rss for item in family))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            stopped.wait(0.1)

    sampler = threading.Thread(target=monitor, daemon=True)
    sampler.start()
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            stdout, stderr = proc.communicate()
        returncode = 124
    finally:
        stopped.set()
        sampler.join()
    return returncode, stdout, stderr, timed_out, {"max_rss_kb": str(peak_rss // 1024)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps", default=",".join(INSTANCES))
    parser.add_argument("--densities", default="1,5,10")
    parser.add_argument("--scenarios", default="0-4")
    parser.add_argument("--algorithms", default="lacam,pibt")
    parser.add_argument("--time-limit", type=int, default=60)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--output-dir", default="results/classic_baselines")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--lacam", default=str(Path.home() / "mapf-baselines/lacam/build/main"))
    parser.add_argument("--pibt", default=str(Path.home() / "mapf-baselines/pibt2/build/mapf"))
    args = parser.parse_args()

    maps = [value for value in args.maps.split(",") if value]
    if not set(maps).issubset(INSTANCES):
        parser.error("unknown map")
    densities = parse_int_list(args.densities, {1, 5, 10, 20, 30, 40, 50, 60})
    scenarios = parse_int_list(args.scenarios, set(range(10)))
    algorithms = [value for value in args.algorithms.split(",") if value]
    if not set(algorithms).issubset({"lacam", "pibt"}):
        parser.error("algorithms must be lacam and/or pibt")
    binaries = {"lacam": Path(args.lacam), "pibt": Path(args.pibt)}
    for algorithm in algorithms:
        if not binaries[algorithm].is_file():
            parser.error(f"missing {algorithm} binary: {binaries[algorithm]}")

    output = ROOT / args.output_dir
    instance_root = output / "instances"
    records = output / "records"
    raw = output / "raw"
    jobs = []
    for map_name in maps:
        spec = INSTANCES[map_name]
        for density in densities:
            agents = density * spec.tiles // 100
            for scenario in scenarios:
                prepared = prepare_cell(instance_root, map_name, density, agents, scenario)
                for algorithm in algorithms:
                    jobs.append((*prepared, density, agents, scenario, algorithm))

    atomic_json(output / "MANIFEST.json", {
        "semantic_scope": "classic MAPF; unique cyclic goals; stay at goal",
        "not_the_lima_main_task": True,
        "maps": maps, "densities": densities, "scenarios": scenarios,
        "algorithms": algorithms, "time_limit_seconds": args.time_limit,
        "binaries": {name: {"path": str(binaries[name].resolve()),
                            "sha256": sha256(binaries[name].resolve())}
                     for name in algorithms},
        "job_count": len(jobs),
    })

    def run(job):
        tag, map_file, scen_file, pibt_file, density, agents, scenario, algorithm = job
        record = records / f"{tag}_{algorithm}.json"
        if record.exists() and not args.rerun:
            return f"{tag}_{algorithm}", "skipped"
        raw.mkdir(parents=True, exist_ok=True)
        result_file = raw / f"{tag}_{algorithm}.txt"
        result_file.unlink(missing_ok=True)
        if algorithm == "lacam":
            solver_command = [str(binaries[algorithm]), "-i", str(scen_file), "-m", str(map_file),
                              "-N", str(agents), "-s", str(scenario), "-t", str(args.time_limit),
                              "-o", str(result_file)]
        else:
            solver_command = [str(binaries[algorithm]), "-i", str(pibt_file), "-s", "PIBT",
                              "-T", str(args.time_limit * 1000), "-o", str(result_file)]
        command = solver_command
        started = time.time()
        returncode, stdout, stderr, timed_out, resource = run_measured(
            command, args.time_limit + 20)
        result = parse_result(result_file)
        payload = {
            "tag": tag, "map": tag.split("_d", 1)[0], "density_percent": density,
            "agents": agents, "scenario": scenario, "algorithm": algorithm,
            "returncode": returncode, "timed_out": timed_out,
            "runner_wall_seconds": time.time() - started, "result": result,
            "resource": resource,
            "stdout_tail": stdout[-1000:], "stderr_tail": stderr[-1000:], "command": command,
        }
        atomic_json(record, payload)
        return f"{tag}_{algorithm}", "solved" if result.get("solved") == "1" else "unsolved"

    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            tag, status = future.result()
            done += 1
            print(f"[{done:3d}/{len(jobs):3d}] {status:8s} {tag}", flush=True)
    print(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
