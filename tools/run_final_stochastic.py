#!/usr/bin/env python3
"""Run step-bounded stochastic-delay experiments on certified inputs."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from summarize_telemetry import summarize_metrics


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DENSITIES = (10, 20, 30)
DEFAULT_PROBABILITIES = (0.05, 0.10, 0.15, 0.20)
TRACE_SPEC = {
    "name": "counter-hash-v1",
    "key": ["instance_seed", "zero_based_agent_id", "one_based_timestep"],
    "seed_salt": "0xa0761d6478bd642f",
    "agent_multiplier": "0xd2b74407b1ce6e93",
    "timestep_multiplier": "0xca5a826395121157",
    "mixer": "SplitMix64",
    "sample": "top 53 bits / 2^53",
    "rule": "delay a non-wait movement command iff sample < probability",
}


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


def parse_probabilities(text: str) -> list[float]:
    values = sorted({float(item) for item in text.split(",") if item.strip()})
    if not values or any(value <= 0.0 or value > 1.0 for value in values):
        raise argparse.ArgumentTypeError("probabilities must be in (0,1]; p=0 reuses deterministic results")
    return values


def parse_fields(text: str) -> dict[str, str]:
    lines = [line for line in text.splitlines() if line.strip()]
    return dict(re.findall(r"(\w+)=([^\s]+)", lines[-1])) if lines else {}


def parse_resource(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    return dict(re.findall(
        r"^(\w+)=([^\n]+)$",
        path.read_text(encoding="utf-8", errors="replace"), re.MULTILINE))


def probability_tag(probability: float) -> str:
    return f"p{int(round(probability * 100)):02d}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm", required=True,
        choices=("lima", "pibt", "lacam-replan", "primal2"))
    parser.add_argument(
        "--input-manifest",
        default="results/revision_final/certified_inputs_v3/MANIFEST.json",
    )
    parser.add_argument("--maps", default=",".join(INSTANCES))
    parser.add_argument("--densities", default=",".join(map(str, DEFAULT_DENSITIES)))
    parser.add_argument("--scenarios", default="0-9")
    parser.add_argument(
        "--probabilities", default=",".join(map(str, DEFAULT_PROBABILITIES)))
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--jobs", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--trace-root", default="results/revision_final/stochastic_trace_descriptors_v1")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--no-early-stop", action="store_true")
    parser.add_argument(
        "--lima", default="results/revision_final/frozen_artifacts_step_v3/lima")
    parser.add_argument(
        "--primal-python", default=str(Path.home() / "miniconda3/envs/primal2/bin/python"))
    parser.add_argument(
        "--primal-script", default=str(Path.home() / "mapf-baselines/PRIMAL2/run_our_instances.py"))
    parser.add_argument(
        "--primal-model",
        default=str(Path.home() / "mapf-baselines/PRIMAL2/model_primal2_oneshot"))
    parser.add_argument(
        "--primal-stall-steps", type=int, default=0,
        help="PRIMAL2-only consecutive no-completion step cutoff (0 = disabled)")
    parser.add_argument(
        "--pibt-repo", default=str(Path.home() / "mapf-baselines/pibt2"))
    parser.add_argument(
        "--pibt-binary",
        default="results/revision_final/frozen_artifacts_step_v2/pibt")
    parser.add_argument(
        "--lacam-repo", default=str(Path.home() / "mapf-baselines/lacam"))
    parser.add_argument(
        "--lacam-binary",
        default="results/revision_final/frozen_artifacts_step_v2/lacam")
    parser.add_argument("--lacam-max-iterations", type=int, default=100000)
    parser.add_argument(
        "--replan-adapter",
        default=str(Path(__file__).resolve().with_name("stochastic_replan_adapter.py")))
    args = parser.parse_args()
    concurrency = args.jobs or (8 if args.algorithm == "lima" else 2)
    if (concurrency < 1 or args.max_steps < 1 or args.primal_stall_steps < 0
            or args.lacam_max_iterations < 1):
        parser.error("jobs and max-steps must be positive")
    maps = [item.strip() for item in args.maps.split(",") if item.strip()]
    if not maps or not set(maps).issubset(INSTANCES):
        parser.error("unknown or empty map selection")
    densities = parse_int_list(args.densities, set(DEFAULT_DENSITIES))
    scenarios = parse_int_list(args.scenarios, set(range(10)))
    probabilities = parse_probabilities(args.probabilities)
    input_manifest_path = (ROOT / args.input_manifest).resolve()
    certified = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    if certified.get("capacity_formula") != "sum(arm capacities) - longest arm":
        parser.error("input manifest does not use the operational capacity")

    lima = (ROOT / args.lima).resolve()
    primal_python = Path(args.primal_python).resolve()
    primal_script = Path(args.primal_script).resolve()
    primal_model = Path(args.primal_model).resolve()
    primal_module_roots = [primal_script.parent, primal_model.parent]
    pibt_repo = Path(args.pibt_repo).resolve()
    lacam_repo = Path(args.lacam_repo).resolve()
    pibt_binary = (ROOT / args.pibt_binary).resolve()
    lacam_binary = (ROOT / args.lacam_binary).resolve()
    replan_adapter = Path(args.replan_adapter).resolve()
    executables = {
        "lima": [lima],
        "pibt": [Path(sys.executable), replan_adapter, pibt_binary],
        "lacam-replan": [Path(sys.executable), replan_adapter, lacam_binary],
        "primal2": [primal_python, primal_script],
    }[args.algorithm]
    for path in executables:
        if not path.is_file():
            parser.error(f"missing executable or adapter: {path}")
    primal_model_files = None
    if args.algorithm == "primal2":
        if not primal_model.is_dir():
            parser.error(f"missing PRIMAL2 checkpoint directory: {primal_model}")
        checkpoint_files = sorted(
            path for path in primal_model.iterdir()
            if path.name == "checkpoint" or path.name.startswith("model-97500.cptk")
        )
        if not checkpoint_files:
            parser.error(f"missing PRIMAL2 checkpoint files: {primal_model}")
        primal_model_files = {
            path.name: sha256(path) for path in checkpoint_files if path.is_file()
        }
    lima_version = subprocess.run(
        [str(lima), "--version"], cwd=ROOT, capture_output=True, text=True, check=False)
    if lima_version.returncode != 0 or "profile=lima-default" not in lima_version.stdout:
        parser.error("instrumented LIMA binary does not expose the frozen profile")

    output = (ROOT / (args.output_dir or
        f"results/revision_final/stochastic_{args.algorithm}_step_v3")).resolve()
    trace_root = (ROOT / args.trace_root).resolve()
    records, resources, logs, metrics = (
        output / "records", output / "resources", output / "logs", output / "metrics")
    for directory in (records, resources, logs, metrics, trace_root):
        directory.mkdir(parents=True, exist_ok=True)

    jobs: list[dict] = []
    inputs: dict[str, str] = {
        str(input_manifest_path.relative_to(ROOT)): sha256(input_manifest_path)
    }
    trace_descriptors: dict[str, str] = {}
    for map_name in maps:
        map_entry = certified["maps"][map_name]
        map_file = map_entry["map_file"]
        map_path = ROOT / map_file
        if sha256(map_path) != map_entry["map_sha256"]:
            parser.error(f"map hash mismatch: {map_name}")
        inputs[map_file] = sha256(map_path)
        for scenario in scenarios:
            for probability in probabilities:
                trace_tag = f"{map_name}_s{scenario}_{probability_tag(probability)}"
                trace_path = trace_root / f"{trace_tag}.json"
                descriptor = {
                    "schema_version": 1, "map": map_name,
                    "scenario": scenario, "instance_seed": scenario,
                    "probability": probability, "trace_spec": TRACE_SPEC,
                }
                if trace_path.is_file():
                    existing = json.loads(trace_path.read_text(encoding="utf-8"))
                    if existing != descriptor:
                        parser.error(f"trace descriptor mismatch: {trace_path}")
                else:
                    atomic_json(trace_path, descriptor)
                trace_descriptors[trace_tag] = sha256(trace_path)
        for density in densities:
            target = f"d{density:02d}"
            target_entry = map_entry["targets"].get(target)
            if target_entry is None:
                parser.error(f"missing certified target {map_name}/{target}")
            agents = int(target_entry["agents"])
            for scenario in scenarios:
                scenario_entry = target_entry["scenarios"][str(scenario)]
                scenario_file = scenario_entry["scenario_file"]
                certificate_file = scenario_entry["certificate_file"]
                if sha256(ROOT / scenario_file) != scenario_entry["scenario_sha256"]:
                    parser.error(f"scenario hash mismatch: {scenario_file}")
                if sha256(ROOT / certificate_file) != scenario_entry["certificate_sha256"]:
                    parser.error(f"certificate hash mismatch: {certificate_file}")
                certificate = json.loads((ROOT / certificate_file).read_text(encoding="utf-8"))
                validation = certificate["validation"]
                if (
                    validation["capacity_violations"] != 0
                    or not validation["unique_starts"]
                    or not validation["unique_goals"]
                    or not validation["traversable_goals"]
                    or validation["same_agent_start_goal"] != 0
                    or validation["reachable_pairs"] != agents
                ):
                    parser.error(f"capacity certificate failed: {certificate_file}")
                inputs[scenario_file] = scenario_entry["scenario_sha256"]
                inputs[certificate_file] = scenario_entry["certificate_sha256"]
                for probability in probabilities:
                    ptag = probability_tag(probability)
                    trace_tag = f"{map_name}_s{scenario}_{ptag}"
                    jobs.append({
                        "map": map_name, "density": density, "target": target,
                        "agents": agents,
                        "tile_density_percent": target_entry["tile_density_percent"],
                        "capacity_load_percent": target_entry["capacity_load_percent"],
                        "scenario": scenario, "probability": probability,
                        "map_file": map_file,
                        "scenario_file": scenario_file,
                        "scenario_sha256": scenario_entry["scenario_sha256"],
                        "certificate_file": certificate_file,
                        "certificate_sha256": scenario_entry["certificate_sha256"],
                        "trace_descriptor": str((trace_root / f"{trace_tag}.json").relative_to(ROOT)),
                        "trace_descriptor_sha256": trace_descriptors[trace_tag],
                        "tag": f"{map_name}_d{density:02d}_a{agents}_s{scenario}_{ptag}",
                    })

    runner = Path(__file__).resolve()
    fingerprint_payload = {
        "schema_version": 2, "algorithm": args.algorithm,
        "semantic_scope": (
            "one-shot; unique capacity-certified starts and goals; fixed-point-free tasks; "
            "disappear at goal; common stochastic movement delay"
        ),
        "trace_spec": TRACE_SPEC,
        "p0_source": "matching deterministic certified step campaign",
        "executables": {str(path): sha256(path) for path in executables},
        "primal2_inference": (
            {
                "mode": "batch",
                "device": "cpu",
                "stall_steps": args.primal_stall_steps,
                "model": str(primal_model),
                "model_files": primal_model_files,
                "module_roots": [str(path) for path in primal_module_roots],
            }
            if args.algorithm == "primal2" else None
        ),
        "stochastic_replanning": (
            {
                "adapter": str(replan_adapter),
                "policy": "replan from observed positions before the next step after any execution divergence",
                "safe_executor": "iterative dependency cancellation; vertex/edge conflict free",
                "communication_accounting": (
                    "initial task upload plus initial route installation; each replan "
                    "counts active-agent state uploads, while route delivery is counted "
                    "only for agents whose executable suffix changes; cached route "
                    "execution counts zero communication"
                ),
                "planner_search_limit": (
                    args.lacam_max_iterations
                    if args.algorithm == "lacam-replan" else None
                ),
            }
            if args.algorithm in ("pibt", "lacam-replan") else None
        ),
        "lima_version": lima_version.stdout.strip(), "runner_sha256": sha256(runner),
        "maps": maps, "densities": densities, "scenarios": scenarios,
        "probabilities": probabilities,
        "termination_policy": (
            "fixed synchronous execution horizon; PRIMAL2 additionally uses "
            "a step-based no-completion stagnation rule; no wall-clock cutoff"
            if args.algorithm == "primal2"
            else (
                "fixed synchronous execution horizon; event-triggered global "
                "replanning after execution divergence; no wall-clock cutoff"
                if args.algorithm in ("pibt", "lacam-replan")
                else "fixed synchronous execution horizon; no wall-clock cutoff"
            )
        ),
        "max_steps": args.max_steps, "inputs": inputs,
        "early_stop_policy": (
            "none for LIMA; for non-LIMA methods, higher densities skipped after "
            "0 successes at a map/probability/density"
            if not args.no_early_stop else "disabled"
        ),
        "trace_descriptors": trace_descriptors,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")).hexdigest()
    manifest_path = output / "MANIFEST.json"
    if manifest_path.is_file() and any(records.glob("*.json")):
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("experiment_fingerprint") != fingerprint:
            parser.error("output directory contains a different experiment fingerprint")
    atomic_json(manifest_path, {
        **fingerprint_payload, "experiment_fingerprint": fingerprint,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runner": str(runner.relative_to(ROOT)), "job_count": len(jobs),
        "jobs_concurrency": concurrency,
        "host": {"hostname": socket.gethostname(), "platform": platform.platform(),
                 "cpu_count": os.cpu_count()},
    })
    lock = output / ".RUNNING"
    if lock.exists():
        parser.error(f"campaign lock already exists: {lock}")
    lock.write_text(f"pid={os.getpid()}\nstarted_utc={datetime.now(timezone.utc).isoformat()}\n")

    def execution_horizon(job: dict) -> dict:
        return {
            "policy": "fixed_max_steps",
            "max_steps": args.max_steps,
        }

    def run(job: dict) -> tuple[str, str]:
        tag = job["tag"]
        horizon = execution_horizon(job)
        effective_max_steps = horizon["max_steps"]
        record_path = records / f"{tag}_{args.algorithm}.json"
        if record_path.exists() and not args.rerun:
            existing = json.loads(record_path.read_text(encoding="utf-8"))
            return tag, "skipped_solved" if existing.get("solved") else "skipped_unsolved"
        resource_path = resources / f"{tag}_{args.algorithm}.txt"
        log_path = logs / f"{tag}_{args.algorithm}.log"
        resource_path.unlink(missing_ok=True)
        if args.algorithm == "lima":
            solver = [
                str(lima), "--profile", "lima-default", "--mode", "solve",
                "--map", job["map_file"], "--scenario", job["scenario_file"],
                "--agents", str(job["agents"]), "--seed", str(job["scenario"]),
                "--max-steps", str(args.max_steps),
                "--failure-prob", str(job["probability"]),
                "--goal-behavior", "disappear", "--no-trace",
                "--metrics", str(metrics / tag),
            ]
        elif args.algorithm == "primal2":
            solver = [
                str(primal_python), str(primal_script),
                "--map", str((ROOT / job["map_file"]).resolve()),
                "--scen", str((ROOT / job["scenario_file"]).resolve()),
                "-n", str(job["agents"]), "--seed", str(1234 + job["scenario"]),
                "--max-steps", str(effective_max_steps), "--progress-every", "0",
                "--delay-prob", str(job["probability"]),
                "--delay-seed", str(job["scenario"]),
                "--model", str(primal_model),
                "--inference", "batch", "--device", "cpu",
            ]
            if args.primal_stall_steps:
                solver.extend(["--stall-steps", str(args.primal_stall_steps)])
        else:
            planner_name = "pibt" if args.algorithm == "pibt" else "lacam"
            solver = [
                str(Path(sys.executable).resolve()), str(replan_adapter),
                "--algorithm", planner_name,
                "--map", str((ROOT / job["map_file"]).resolve()),
                "--scen", str((ROOT / job["scenario_file"]).resolve()),
                "--agents", str(job["agents"]),
                "--seed", str(job["scenario"]),
                "--max-steps", str(effective_max_steps),
                "--delay-prob", str(job["probability"]),
                "--delay-seed", str(job["scenario"]),
                "--pibt-repo", str(pibt_repo),
                "--pibt-binary", str(pibt_binary),
                "--lacam-repo", str(lacam_repo),
                "--lacam-binary", str(lacam_binary),
                "--lacam-max-iterations", str(args.lacam_max_iterations),
            ]
        command = [
            "/usr/bin/time", "-f",
            "max_rss_kb=%M\nuser_seconds=%U\nsystem_seconds=%S\nelapsed_seconds=%e",
            "-o", str(resource_path), *solver,
        ]
        started = time.time()
        process_env = None
        if args.algorithm == "primal2":
            process_env = os.environ.copy()
            module_paths = [str(path) for path in primal_module_roots]
            if process_env.get("PYTHONPATH"):
                module_paths.append(process_env["PYTHONPATH"])
            process_env["PYTHONPATH"] = os.pathsep.join(module_paths)
        proc = subprocess.Popen(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, start_new_session=True, env=process_env)
        stdout, stderr = proc.communicate()
        returncode = proc.returncode
        result = parse_fields(stdout)
        if args.algorithm == "primal2" and "completed" in result:
            done, total = result["completed"].split("/", 1)
            result["solved"] = "1" if done == total else "0"
        solved = (
            result.get("status") == "completed" if args.algorithm == "lima"
            else result.get("solved") == "1")
        telemetry = summarize_metrics(metrics / tag) if args.algorithm == "lima" else None
        if args.algorithm == "lima":
            solved = (
                returncode == 0 and solved
                and telemetry["path_conformity"]["online_validation_ok"]
            )
        elif args.algorithm in ("pibt", "lacam-replan"):
            telemetry = None
            solved = (
                returncode == 0 and solved
                and result.get("vertex_conflicts") == "0"
                and result.get("edge_conflicts") == "0"
            )
        log_path.write_text(stdout + ("\n[stderr]\n" + stderr if stderr else ""), encoding="utf-8")
        atomic_json(record_path, {
            **job, "algorithm": args.algorithm, "returncode": returncode,
            "timed_out": False, "solved": solved,
            "runner_wall_seconds": time.time() - started,
            "result": result, "resource": parse_resource(resource_path),
            "telemetry": telemetry,
            "execution_horizon": horizon,
            "command": command, "log": str(log_path.relative_to(ROOT)),
            "experiment_fingerprint": fingerprint,
        })
        return tag, "solved" if solved else "unsolved"

    completed = 0

    def skip_after_zero(job: dict, source_density: int) -> tuple[str, str]:
        tag = job["tag"]
        record_path = records / f"{tag}_{args.algorithm}.json"
        if record_path.exists() and not args.rerun:
            existing = json.loads(record_path.read_text(encoding="utf-8"))
            return tag, "skipped_solved" if existing.get("solved") else "skipped_unsolved"
        atomic_json(record_path, {
            **job,
            "algorithm": args.algorithm,
            "returncode": None,
            "timed_out": False,
            "solved": False,
            "status": "early_stopped_after_zero_success",
            "early_stop_source_density": source_density,
            "runner_wall_seconds": 0.0,
            "result": {},
            "resource": {},
            "execution_horizon": execution_horizon(job),
            "command": None,
            "log": None,
            "experiment_fingerprint": fingerprint,
        })
        return tag, "early_stop"

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            for map_name in maps:
                for probability in probabilities:
                    stop_source = None
                    for density in densities:
                        group = [
                            job for job in jobs
                            if job["map"] == map_name
                            and job["probability"] == probability
                            and job["density"] == density
                        ]
                        if stop_source is not None:
                            outcomes = [skip_after_zero(job, stop_source) for job in group]
                        else:
                            futures = [pool.submit(run, job) for job in group]
                            outcomes = [future.result()
                                        for future in concurrent.futures.as_completed(futures)]
                        successes = 0
                        for tag, status in outcomes:
                            completed += 1
                            if status in ("solved", "skipped_solved"):
                                successes += 1
                            print(
                                f"[{completed:3d}/{len(jobs):3d}] {status:15s} {tag}",
                                flush=True,
                            )
                        if (
                            args.algorithm != "lima"
                            and not args.no_early_stop
                            and stop_source is None
                            and successes == 0
                        ):
                            stop_source = density
                            print(
                                f"early-stop {map_name} p={probability}: "
                                f"0/{len(group)} successes at d{density:02d}",
                                flush=True,
                            )
    finally:
        lock.unlink(missing_ok=True)
    print(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
