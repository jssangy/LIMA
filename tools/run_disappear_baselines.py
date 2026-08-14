#!/usr/bin/env python3
"""Run disappear-at-target baselines on the exact submitted LIMA instances."""

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
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Instance:
    map_file: str
    scenario_template: str
    tiles: int


INSTANCES = {
    "warehouse_10_20": Instance(
        "data/maps/warehouse_10_20_paper.map",
        "data/scenarios/warehouse-10-20-paper/warehouse-10-20-paper_s{s}.scen", 2649),
    "warehouse_20_40": Instance(
        "data/maps/warehouse_20_40_paper.map",
        "data/scenarios/warehouse-20-40-paper/warehouse-20-40-paper_s{s}.scen", 10499),
    "cross_3030": Instance(
        "data/maps/cross_3030_paper.map",
        "data/scenarios/cross-30-30-paper/cross-30-30-paper_s{s}.scen", 10200),
}


def int_list(text: str, allowed: set[int]) -> list[int]:
    values: set[int] = set()
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            lo, hi = (int(v) for v in item.split("-", 1))
            values.update(range(lo, hi + 1))
        else:
            values.add(int(item))
    if not values.issubset(allowed):
        raise argparse.ArgumentTypeError(f"values must be a subset of {sorted(allowed)}")
    return sorted(values)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        tmp = Path(stream.name)
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_fields(text: str) -> dict[str, str]:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return {}
    return dict(re.findall(r"(\w+)=([^\s]+)", lines[-1]))


def parse_resource(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    return dict(re.findall(r"^(\w+)=([^\n]+)$",
                           path.read_text(encoding="utf-8", errors="replace"), re.MULTILINE))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithms", default="cbs,primal2")
    parser.add_argument("--maps", default=",".join(INSTANCES))
    parser.add_argument("--densities", default="1,5")
    parser.add_argument("--scenarios", default="0-4")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--cbs-time-limit", type=int, default=60)
    parser.add_argument("--primal-time-limit", type=int, default=900)
    parser.add_argument("--output-dir", default="results/disappear_baselines")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--cbs", default="build_gating2/cbs_baseline")
    parser.add_argument("--primal-python", default=str(Path.home() / "miniconda3/envs/primal2/bin/python"))
    parser.add_argument("--primal-script", default=str(Path.home() / "mapf-baselines/PRIMAL2/run_our_instances.py"))
    args = parser.parse_args()

    algorithms = [v for v in args.algorithms.split(",") if v]
    if not set(algorithms).issubset({"cbs", "primal2"}):
        parser.error("algorithms must be cbs and/or primal2")
    maps = [v for v in args.maps.split(",") if v]
    if not set(maps).issubset(INSTANCES):
        parser.error("unknown map")
    densities = int_list(args.densities, {1, 5, 10, 20, 30, 40, 50, 60})
    scenarios = int_list(args.scenarios, set(range(10)))
    cbs = (ROOT / args.cbs).resolve()
    primal_python = Path(args.primal_python)
    primal_script = Path(args.primal_script)
    if "cbs" in algorithms and not cbs.is_file():
        parser.error(f"missing CBS binary: {cbs}")
    if "primal2" in algorithms and (not primal_python.is_file() or not primal_script.is_file()):
        parser.error("missing PRIMAL2 runtime or script")

    output = ROOT / args.output_dir
    records = output / "records"
    raw = output / "raw"
    resources = output / "resources"
    raw.mkdir(parents=True, exist_ok=True)
    resources.mkdir(parents=True, exist_ok=True)
    jobs = []
    for map_name in maps:
        spec = INSTANCES[map_name]
        for density in densities:
            agents = density * spec.tiles // 100
            for scenario in scenarios:
                tag = f"{map_name}_d{density:02d}_a{agents}_s{scenario}"
                for algorithm in algorithms:
                    jobs.append((map_name, density, agents, scenario, tag, algorithm, spec))
    atomic_json(output / "MANIFEST.json", {
        "semantic_scope": "submitted repeated-sink task; disappear at target",
        "algorithms": algorithms, "maps": maps, "densities": densities,
        "scenarios": scenarios, "cbs_time_limit_seconds": args.cbs_time_limit,
        "primal_time_limit_seconds": args.primal_time_limit, "job_count": len(jobs),
        "executables": {
            "cbs": {"path": str(cbs), "sha256": sha256(cbs)} if "cbs" in algorithms else None,
            "primal_python": {"path": str(primal_python), "sha256": sha256(primal_python)}
            if "primal2" in algorithms else None,
            "primal_script": {"path": str(primal_script), "sha256": sha256(primal_script)}
            if "primal2" in algorithms else None,
        },
    })

    def run(job):
        map_name, density, agents, scenario, tag, algorithm, spec = job
        record = records / f"{tag}_{algorithm}.json"
        if record.exists() and not args.rerun:
            return f"{tag}_{algorithm}", "skipped"
        map_path = (ROOT / spec.map_file).resolve()
        scenario_path = (ROOT / spec.scenario_template.format(s=scenario)).resolve()
        if algorithm == "cbs":
            solver_command = [str(cbs), "--map", str(map_path), "--scenario", str(scenario_path),
                              "--agents", str(agents), "--time-limit", str(args.cbs_time_limit)]
            timeout = args.cbs_time_limit + 20
        else:
            solver_command = [str(primal_python), str(primal_script), "--map", str(map_path),
                              "--scen", str(scenario_path), "-n", str(agents),
                              "--seed", str(1234 + scenario), "--progress-every", "0"]
            timeout = args.primal_time_limit
        resource_file = resources / f"{tag}_{algorithm}.txt"
        resource_file.unlink(missing_ok=True)
        command = ["/usr/bin/time", "-f", "max_rss_kb=%M\nuser_seconds=%U\nsystem_seconds=%S",
                   "-o", str(resource_file), *solver_command]
        started = time.time()
        timed_out = False
        proc = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, start_new_session=True)
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
        log_path = raw / f"{tag}_{algorithm}.log"
        log_path.write_text(stdout + ("\n[stderr]\n" + stderr if stderr else ""), encoding="utf-8")
        result = parse_fields(stdout)
        if algorithm == "primal2" and "completed" in result:
            done, total = result["completed"].split("/", 1)
            result["solved"] = "1" if done == total else "0"
            result["makespan"] = result.get("steps", "")
            result["elapsed_s"] = result.get("wall_s", "")
        payload = {
            "tag": tag, "map": map_name, "density_percent": density,
            "agents": agents, "scenario": scenario, "algorithm": algorithm,
            "returncode": returncode, "timed_out": timed_out,
            "runner_wall_seconds": time.time() - started, "result": result,
            "resource": parse_resource(resource_file),
            "command": command, "log": str(log_path.relative_to(ROOT)),
        }
        atomic_json(record, payload)
        return f"{tag}_{algorithm}", "solved" if result.get("solved") == "1" else "unsolved"

    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            tag, status = future.result(); done += 1
            print(f"[{done:3d}/{len(jobs):3d}] {status:8s} {tag}", flush=True)
    print(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
