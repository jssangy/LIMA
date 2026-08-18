#!/usr/bin/env python3
"""Run step-bounded one-shot experiments on capacity-certified inputs.

Wall time is measured but never used as a success or termination criterion.
All execution methods share the same discrete simulation horizon. CBS uses a
deterministic high-level expansion bound for its offline search.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from summarize_telemetry import summarize_metrics


ROOT = Path(__file__).resolve().parent.parent
FREEZE_MANIFEST = ROOT / "results/revision_final/frozen_artifacts_step_v3/MANIFEST.json"


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


def parse_fields(text: str) -> dict[str, str]:
    lines = [line for line in text.splitlines() if line.strip()]
    return dict(re.findall(r"(\w+)=([^\s]+)", lines[-1])) if lines else {}


def parse_resource(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    return dict(re.findall(
        r"^(\w+)=([^\n]+)$",
        path.read_text(encoding="utf-8", errors="replace"), re.MULTILINE))


def parse_key_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    return dict(re.findall(
        r"^(\w+)=(.*)$",
        path.read_text(encoding="utf-8", errors="replace"), re.MULTILINE))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm", required=True,
        choices=("lima", "cbs", "lacam", "pibt", "primal2"))
    parser.add_argument(
        "--input-manifest",
        default="results/revision_final/certified_inputs_v3/MANIFEST.json")
    parser.add_argument("--maps", help="optional comma-separated map filter")
    parser.add_argument("--targets", help="optional comma-separated target-label filter")
    parser.add_argument("--scenarios", help="optional comma-separated scenario-index filter")
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--cbs-max-expansions", type=int, default=100000)
    parser.add_argument("--lacam-max-iterations", type=int, default=100000)
    parser.add_argument(
        "--lima-binary", default="results/revision_final/frozen_artifacts_step_v3/lima")
    parser.add_argument(
        "--cbs-binary", default="results/revision_final/frozen_artifacts_step_v2/cbs_baseline")
    parser.add_argument("--pibt-repo", default=str(Path.home() / "mapf-baselines/pibt2"))
    parser.add_argument(
        "--pibt-binary",
        default=str(ROOT / "results/revision_final/frozen_artifacts_step_v2/pibt"))
    parser.add_argument("--lacam-repo", default=str(Path.home() / "mapf-baselines/lacam"))
    parser.add_argument(
        "--lacam-binary",
        default=str(ROOT / "results/revision_final/frozen_artifacts_step_v2/lacam"))
    parser.add_argument("--jobs", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--reuse-records-from", action="append", default=[],
        help="campaign directory whose solved records may be reused with provenance")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument(
        "--no-early-stop", action="store_true",
        help="continue higher densities after a 0-success density",
    )
    parser.add_argument("--freeze-manifest", default=str(FREEZE_MANIFEST))
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
    args = parser.parse_args()
    jobs_concurrency = args.jobs or (8 if args.algorithm == "lima" else 2)
    if (jobs_concurrency < 1 or args.max_steps < 1 or args.cbs_max_expansions < 1
            or args.lacam_max_iterations < 1 or args.primal_stall_steps < 0):
        parser.error("jobs and deterministic step/search limits must be positive")

    freeze_path = Path(args.freeze_manifest).resolve()
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "frozen":
        parser.error("freeze manifest is not frozen")
    lima_artifact = freeze["artifacts"]["lima"]
    lima = (ROOT / args.lima_binary).resolve()
    cbs = (ROOT / args.cbs_binary).resolve()
    lacam_repo = Path(args.lacam_repo).resolve()
    lacam = (lacam_repo / args.lacam_binary).resolve()
    pibt_repo = Path(args.pibt_repo).resolve()
    pibt = (pibt_repo / args.pibt_binary).resolve()
    primal_python = Path(args.primal_python).resolve()
    primal_script = Path(args.primal_script).resolve()
    primal_model = Path(args.primal_model).resolve()
    primal_module_roots = [primal_script.parent, primal_model.parent]
    executables = {
        "lima": [lima], "cbs": [cbs], "lacam": [lacam], "pibt": [pibt],
        "primal2": [primal_python, primal_script]
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
        [str(lima), "--version"], cwd=ROOT,
        capture_output=True, text=True, check=False)
    if lima_version.returncode != 0 or "profile=lima-default" not in lima_version.stdout:
        parser.error("instrumented LIMA binary does not expose the frozen profile")
    pibt_provenance = None
    if args.algorithm == "pibt":
        pibt_help = subprocess.run(
            [str(pibt), "--help"], cwd=pibt_repo,
            capture_output=True, text=True, check=False)
        if ("--disappear-at-goal" not in pibt_help.stdout
                or "--exclusive-boundary-goals" not in pibt_help.stdout):
            parser.error(
                "PIBT binary lacks the disappear/exclusive-boundary adapter")
        pibt_commit = subprocess.run(
            ["git", "-C", str(pibt_repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
        pibt_diff = subprocess.run(
            ["git", "-C", str(pibt_repo), "diff", "--binary"],
            capture_output=True, check=True).stdout
        pibt_provenance = {
            "repository": str(pibt_repo),
            "commit": pibt_commit,
            "working_diff_sha256": hashlib.sha256(pibt_diff).hexdigest(),
        }
    lacam_provenance = None
    if args.algorithm == "lacam":
        lacam_help = subprocess.run(
            [str(lacam), "--help"], cwd=lacam_repo,
            capture_output=True, text=True, check=False)
        if ("--disappear-at-goal" not in lacam_help.stdout
                or "--exclusive-boundary-goals" not in lacam_help.stdout
                or "--max_iterations" not in lacam_help.stdout):
            parser.error(
                "LaCAM binary lacks the disappear/exclusive-boundary/search-limit adapter")
        lacam_commit = subprocess.run(
            ["git", "-C", str(lacam_repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
        lacam_diff = subprocess.run(
            ["git", "-C", str(lacam_repo), "diff", "--binary"],
            capture_output=True, check=True).stdout
        lacam_provenance = {
            "repository": str(lacam_repo),
            "commit": lacam_commit,
            "working_diff_sha256": hashlib.sha256(lacam_diff).hexdigest(),
        }

    if args.algorithm == "cbs":
        cbs_help = subprocess.run(
            [str(cbs)], cwd=ROOT, capture_output=True, text=True, check=False)
        cbs_usage = cbs_help.stdout + cbs_help.stderr
        if "--exclusive-boundary-goals" not in cbs_usage:
            parser.error("CBS binary lacks the exclusive-boundary adapter")

    input_manifest_path = (ROOT / args.input_manifest).resolve()
    inputs = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    if inputs.get("capacity_formula") != "sum(arm capacities) - longest arm":
        parser.error("certified input manifest does not use the operational capacity")
    boundary_exit_semantics = (
        inputs.get("goal_semantics", {}).get("type")
        == "physical boundary service event"
        and inputs.get("goal_semantics", {}).get("completion")
        == "agent disappears on first arrival at its assigned G workstation"
    )
    if boundary_exit_semantics and "boundary_goals=exclusive-v1" not in lima_version.stdout:
        parser.error("LIMA binary lacks the exclusive assigned-G entry policy")
    if boundary_exit_semantics and args.algorithm == "primal2":
        parser.error(
            f"{args.algorithm} adapter does not implement disappearing shared-exit goals; "
            "do not run it on the managed-boundary mission"
        )
    map_filter = set(args.maps.split(",")) if args.maps else None
    target_filter = set(args.targets.split(",")) if args.targets else None
    scenario_filter = set(map(int, args.scenarios.split(","))) if args.scenarios else None
    cells: list[dict] = []
    for map_name, map_entry in inputs["maps"].items():
        if map_filter is not None and map_name not in map_filter:
            continue
        map_path = ROOT / map_entry["map_file"]
        if sha256(map_path) != map_entry["map_sha256"]:
            parser.error(f"map hash mismatch: {map_name}")
        for target, target_entry in map_entry["targets"].items():
            if target_filter is not None and target not in target_filter:
                continue
            agents = int(target_entry["agents"])
            for scenario_text, entry in target_entry["scenarios"].items():
                scenario = int(scenario_text)
                if scenario_filter is not None and scenario not in scenario_filter:
                    continue
                scenario_path = ROOT / entry["scenario_file"]
                certificate_path = ROOT / entry["certificate_file"]
                if sha256(scenario_path) != entry["scenario_sha256"]:
                    parser.error(f"scenario hash mismatch: {scenario_path}")
                if sha256(certificate_path) != entry["certificate_sha256"]:
                    parser.error(f"certificate hash mismatch: {certificate_path}")
                certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
                validation = certificate["validation"]
                if (
                    validation["capacity_violations"] != 0
                    or not validation["unique_starts"]
                    or not validation["traversable_goals"]
                    or validation["same_agent_start_goal"] != 0
                    or validation["reachable_pairs"] != agents
                ):
                    parser.error(f"capacity certificate failed: {certificate_path}")
                if boundary_exit_semantics:
                    if (
                        not validation.get("physical_boundary_goals")
                        or not validation.get("disappear_at_goal")
                        or not validation.get("exclusive_assigned_boundary_entry")
                        or validation.get("persistent_goal_occupancy_required") is not False
                    ):
                        parser.error(
                            f"boundary-exit certificate failed: {certificate_path}")
                elif not validation["unique_goals"]:
                    parser.error(f"persistent one-shot goals are not unique: {certificate_path}")
                cells.append({
                    "map": map_name, "target": target, "agents": agents,
                    "tile_density_percent": target_entry.get(
                        "tile_density_percent", 100.0 * agents / map_entry["tiles"]),
                    "capacity_load_percent": target_entry.get(
                        "capacity_load_percent",
                        100.0 * agents / map_entry["maximum_agents"]),
                    "scenario": scenario, "map_file": map_entry["map_file"],
                    "scenario_file": entry["scenario_file"],
                    "scenario_sha256": entry["scenario_sha256"],
                    "certificate_file": entry["certificate_file"],
                    "certificate_sha256": entry["certificate_sha256"],
                    "tag": f"{map_name}_{target}_a{agents}_s{scenario}",
                })
    if not cells:
        parser.error("certified input filters selected no cells")
    cells.sort(key=lambda cell: (
        cell["map"], cell["tile_density_percent"], cell["agents"], cell["scenario"]))

    output = (ROOT / (args.output_dir or
        f"results/revision_final/oneshot_{args.algorithm}_certified_step_v3")).resolve()
    reuse_roots = [(ROOT / path).resolve() for path in args.reuse_records_from]
    reuse_sources = []
    for source in reuse_roots:
        manifest = source / "MANIFEST.json"
        if not manifest.is_file() or not (source / "records").is_dir():
            parser.error(f"invalid reuse campaign: {source}")
        reuse_sources.append({
            "campaign": str(source.relative_to(ROOT)),
            "manifest_sha256": sha256(manifest),
        })
    records, resources, logs, metrics, instances, solutions = (
        output / "records", output / "resources", output / "logs", output / "metrics",
        output / "instances", output / "solutions")
    for directory in (records, resources, logs, metrics, instances, solutions):
        directory.mkdir(parents=True, exist_ok=True)

    pibt_map_names: dict[str, str] = {}
    if args.algorithm == "pibt":
        map_cache = pibt_repo / "map"
        map_cache.mkdir(parents=True, exist_ok=True)
        for cell in cells:
            if cell["map"] in pibt_map_names:
                continue
            source = ROOT / cell["map_file"]
            cache_name = f"lima_{sha256(source)[:16]}.map"
            cached = map_cache / cache_name
            if cached.exists() and sha256(cached) != sha256(source):
                parser.error(f"PIBT map-cache hash mismatch: {cached}")
            if not cached.exists():
                temporary = cached.with_suffix(".tmp")
                shutil.copyfile(source, temporary)
                os.replace(temporary, cached)
            pibt_map_names[cell["map"]] = cache_name

    runner = Path(__file__).resolve()
    fingerprint_payload = {
        "schema_version": 3,
        "semantic_scope": (
            "one-shot; capacity-certified unique starts; repeated physical G exits "
            "are service events; an active agent may enter only its assigned G; "
            "agent disappears on first arrival; every other active position is "
            "inside the managed-cell union"
            if boundary_exit_semantics else
            "one-shot; unique capacity-certified starts and goals; "
            "fixed-point-free tasks; disappear at goal"
        ),
        "boundary_exit_semantics": boundary_exit_semantics,
        "boundary_entry_policy": (
            "exclusive assigned-G entry; atomic disappear on arrival"
            if boundary_exit_semantics else None
        ),
        "movement_domain": inputs.get("movement_domain"),
        "goal_semantics": inputs.get("goal_semantics"),
        "algorithm": args.algorithm,
        "executables": {str(path): sha256(path) for path in executables},
        "lima_version": lima_version.stdout.strip(),
        "reference_lima_sha256": lima_artifact["sha256"],
        "runner_sha256": sha256(runner),
        "input_manifest_sha256": sha256(input_manifest_path),
        "cells": [cell["tag"] for cell in cells],
        "termination_policy": (
            "fixed synchronous execution horizon; PRIMAL2 additionally uses "
            "a step-based no-completion stagnation rule; no wall-clock cutoff"
            if args.algorithm == "primal2"
            else "fixed synchronous execution horizon; no wall-clock cutoff"
        ),
        "max_steps": args.max_steps,
        "cbs_max_expansions": args.cbs_max_expansions,
        "lacam_max_iterations": args.lacam_max_iterations,
        "lacam_provenance": lacam_provenance,
        "pibt_provenance": pibt_provenance,
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
        "early_stop_policy": (
            "none for LIMA; higher densities skipped after 0 successes at a density"
            if not args.no_early_stop else "disabled"
        ),
        "reuse_sources": reuse_sources,
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
        "runner": str(runner.relative_to(ROOT)),
        "input_manifest": str(input_manifest_path.relative_to(ROOT)),
        "freeze_manifest": str(freeze_path.relative_to(ROOT)),
        "job_count": len(cells), "jobs_concurrency": jobs_concurrency,
        "host": {"hostname": socket.gethostname(), "platform": platform.platform(),
                 "cpu_count": os.cpu_count()},
    })
    lock = output / ".RUNNING"
    if lock.exists():
        parser.error(f"campaign lock already exists: {lock}")
    lock.write_text(f"pid={os.getpid()}\nstarted_utc={datetime.now(timezone.utc).isoformat()}\n")

    def execution_horizon(cell: dict) -> dict:
        return {
            "policy": "fixed_max_steps",
            "max_steps": args.max_steps,
        }

    def materialize_reusable(cell: dict, record_path: Path) -> bool:
        if args.rerun:
            return False
        horizon = execution_horizon(cell)
        for source in reuse_roots:
            source_path = source / "records" / record_path.name
            if not source_path.is_file():
                continue
            reusable = json.loads(source_path.read_text(encoding="utf-8"))
            result = reusable.get("result") or {}
            try:
                result_steps = int(result.get("steps", horizon["max_steps"] + 1))
            except (TypeError, ValueError):
                continue
            if (
                reusable.get("algorithm") == args.algorithm
                and reusable.get("tag") == cell["tag"]
                and reusable.get("solved") is True
                and reusable.get("returncode") == 0
                and result.get("completed") == f"{cell['agents']}/{cell['agents']}"
                and result_steps <= horizon["max_steps"]
                and reusable.get("scenario_sha256") == cell["scenario_sha256"]
                and reusable.get("certificate_sha256") == cell["certificate_sha256"]
            ):
                atomic_json(record_path, {
                    **reusable,
                    "experiment_fingerprint": fingerprint,
                    "execution_horizon": horizon,
                    "reused_record": {
                        "source": str(source_path.relative_to(ROOT)),
                        "source_sha256": sha256(source_path),
                        "source_experiment_fingerprint": reusable.get(
                            "experiment_fingerprint"),
                        "reason": (
                            "validated solved record completed within the active "
                            "fixed execution horizon"
                        ),
                    },
                })
                return True
        return False

    if reuse_roots and not args.rerun:
        for cell in cells:
            record_path = records / f"{cell['tag']}_{args.algorithm}.json"
            if not record_path.exists():
                materialize_reusable(cell, record_path)

    def run(cell: dict) -> tuple[str, str]:
        tag = cell["tag"]
        horizon = execution_horizon(cell)
        effective_max_steps = horizon["max_steps"]
        record_path = records / f"{tag}_{args.algorithm}.json"
        if record_path.exists() and not args.rerun:
            existing = json.loads(record_path.read_text(encoding="utf-8"))
            return tag, "skipped_solved" if existing.get("solved") else "skipped_unsolved"
        if materialize_reusable(cell, record_path):
            return tag, "reused_solved"
        resource_path = resources / f"{tag}_{args.algorithm}.txt"
        log_path = logs / f"{tag}_{args.algorithm}.log"
        resource_path.unlink(missing_ok=True)
        if args.algorithm == "lima":
            solver = [
                str(lima), "--profile", "lima-default", "--mode", "solve",
                "--map", cell["map_file"], "--scenario", cell["scenario_file"],
                "--agents", str(cell["agents"]), "--seed", str(cell["scenario"]),
                "--max-steps", str(args.max_steps),
                "--stall-threshold", str(args.max_steps + 1),
                "--goal-behavior", "disappear",
                "--no-trace",
                "--metrics", str(metrics / tag),
            ]
            if boundary_exit_semantics:
                solver.append("--exclusive-boundary-goals")
        elif args.algorithm == "cbs":
            solver = [
                str(cbs), "--map", cell["map_file"], "--scenario", cell["scenario_file"],
                "--agents", str(cell["agents"]),
                "--max-expansions", str(args.cbs_max_expansions),
            ]
            if boundary_exit_semantics:
                solver.append("--exclusive-boundary-goals")
        elif args.algorithm == "pibt":
            instance_path = instances / f"{tag}.txt"
            solution_path = solutions / f"{tag}.txt"
            scenario_rows = [
                line.split()
                for line in (ROOT / cell["scenario_file"]).read_text(
                    encoding="utf-8").splitlines()[1:]
                if line.strip()
            ]
            if len(scenario_rows) != cell["agents"]:
                raise ValueError(f"PIBT scenario size mismatch: {cell['scenario_file']}")
            instance_lines = [
                f"map_file={pibt_map_names[cell['map']]}",
                f"agents={cell['agents']}",
                f"seed={cell['scenario']}",
                "random_problem=0",
                f"max_timestep={args.max_steps}",
                "max_comp_time=2147483647",
            ]
            instance_lines.extend(
                f"{row[4]},{row[5]},{row[6]},{row[7]}"
                for row in scenario_rows
            )
            instance_path.write_text("\n".join(instance_lines) + "\n", encoding="utf-8")
            solver = [
                str(pibt), "--instance", str(instance_path),
                "--solver", "PIBT", "--output", str(solution_path),
                "--log-short", "--disappear-at-goal",
            ]
            if boundary_exit_semantics:
                solver.append("--exclusive-boundary-goals")
        elif args.algorithm == "lacam":
            solution_path = solutions / f"{tag}.txt"
            solver = [
                str(lacam), "--map", str((ROOT / cell["map_file"]).resolve()),
                "--scen", str((ROOT / cell["scenario_file"]).resolve()),
                "--num", str(cell["agents"]), "--seed", str(cell["scenario"]),
                "--time_limit_sec", "0",
                "--max_iterations", str(args.lacam_max_iterations),
                "--disappear-at-goal", "--log_short",
                "--output", str(solution_path),
            ]
            if boundary_exit_semantics:
                solver.append("--exclusive-boundary-goals")
        else:
            solver = [
                str(primal_python), str(primal_script),
                "--map", str((ROOT / cell["map_file"]).resolve()),
                "--scen", str((ROOT / cell["scenario_file"]).resolve()),
                "-n", str(cell["agents"]), "--seed", str(1234 + cell["scenario"]),
                "--max-steps", str(effective_max_steps), "--progress-every", "0",
                "--model", str(primal_model),
                "--inference", "batch", "--device", "cpu",
            ]
            if args.primal_stall_steps:
                solver.extend(["--stall-steps", str(args.primal_stall_steps)])
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
        stdout_result = parse_fields(stdout)
        result = stdout_result
        if args.algorithm in ("pibt", "lacam"):
            result = parse_key_file(solution_path)
            if args.algorithm == "pibt" and "completed" in stdout_result:
                result["completed"] = stdout_result["completed"]
        if args.algorithm == "primal2" and "completed" in result:
            done, total = result["completed"].split("/", 1)
            result["solved"] = "1" if done == total else "0"
            result["makespan"] = result.get("steps", "")
            result["elapsed_s"] = result.get("wall_s", "")
        if args.algorithm == "lima":
            telemetry = summarize_metrics(metrics / tag)
            solved = (
                returncode == 0
                and result.get("status") == "completed"
                and telemetry["path_conformity"]["online_validation_ok"]
            )
        elif args.algorithm in ("cbs", "lacam", "pibt"):
            telemetry = None
            solved = (
                result.get("solved") == "1"
                and int(result.get("makespan", args.max_steps + 1)) <= args.max_steps
                and (not boundary_exit_semantics
                     or result.get("boundary_entry_violations") == "0")
            )
        else:
            telemetry = None
            solved = result.get("solved") == "1"
        log_path.write_text(stdout + ("\n[stderr]\n" + stderr if stderr else ""), encoding="utf-8")
        atomic_json(record_path, {
            **cell, "algorithm": args.algorithm, "returncode": returncode,
            "timed_out": False, "solved": solved,
            "runner_wall_seconds": time.time() - started,
            "result": result, "resource": parse_resource(resource_path),
            "telemetry": telemetry,
            "execution_horizon": horizon,
            "command": command, "log": str(log_path.relative_to(ROOT)),
            "experiment_fingerprint": fingerprint,
        })
        if solved:
            status = "solved"
        elif args.algorithm == "cbs" and int(result.get("expansions", 0)) >= args.cbs_max_expansions:
            status = "search_limit"
        elif (args.algorithm == "lacam"
              and int(result.get("search_iterations", 0)) >= args.lacam_max_iterations):
            status = "search_limit"
        else:
            status = "unsolved"
        return tag, status

    completed = 0

    def skip_after_zero(cell: dict, source_target: str) -> tuple[str, str]:
        tag = cell["tag"]
        record_path = records / f"{tag}_{args.algorithm}.json"
        if record_path.exists() and not args.rerun:
            existing = json.loads(record_path.read_text(encoding="utf-8"))
            return tag, "skipped_solved" if existing.get("solved") else "skipped_unsolved"
        atomic_json(record_path, {
            **cell,
            "algorithm": args.algorithm,
            "returncode": None,
            "timed_out": False,
            "solved": False,
            "status": "early_stopped_after_zero_success",
            "early_stop_source_target": source_target,
            "runner_wall_seconds": 0.0,
            "result": {},
            "resource": {},
            "execution_horizon": execution_horizon(cell),
            "command": None,
            "log": None,
            "experiment_fingerprint": fingerprint,
        })
        return tag, "early_stop"

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs_concurrency) as pool:
            by_map: dict[str, list[tuple[str, list[dict]]]] = {}
            for cell in cells:
                groups = by_map.setdefault(cell["map"], [])
                if not groups or groups[-1][0] != cell["target"]:
                    groups.append((cell["target"], []))
                groups[-1][1].append(cell)
            for map_name, groups in by_map.items():
                stop_source = None
                for target, group in groups:
                    if stop_source is not None:
                        outcomes = [skip_after_zero(cell, stop_source) for cell in group]
                    else:
                        futures = [pool.submit(run, cell) for cell in group]
                        outcomes = [future.result()
                                    for future in concurrent.futures.as_completed(futures)]
                    successes = 0
                    for tag, status in outcomes:
                        completed += 1
                        if status in ("solved", "skipped_solved", "reused_solved"):
                            successes += 1
                        print(
                            f"[{completed:3d}/{len(cells):3d}] {status:15s} {tag}",
                            flush=True,
                        )
                    if (
                        args.algorithm != "lima"
                        and not args.no_early_stop
                        and stop_source is None
                        and successes == 0
                    ):
                        stop_source = target
                        print(
                            f"early-stop {map_name}: 0/{len(group)} successes at {target}",
                            flush=True,
                        )
    finally:
        lock.unlink(missing_ok=True)
    print(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
