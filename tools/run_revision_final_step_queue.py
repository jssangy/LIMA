#!/usr/bin/env python3
"""Run the optimized, step-bounded LIMA revision experiment continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_MANIFEST = (
    ROOT / "results/revision_final/frozen_artifacts_step_v4_optimized_r2/MANIFEST.json"
)
QUEUE_ROOT = ROOT / "results/revision_final/queue_step_v4_optimized_r1"


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


def verify_frozen_artifacts(manifest: dict) -> None:
    if manifest.get("status") != "frozen":
        raise RuntimeError("artifact manifest is not frozen")
    for name in ("lima", "cbs_baseline", "lacam", "pibt"):
        artifact = manifest["artifacts"][name]
        path = ROOT / artifact["path"]
        if not path.is_file() or sha256(path) != artifact["sha256"]:
            raise RuntimeError(f"frozen artifact mismatch: {name}")
    primal2 = manifest["artifacts"]["primal2_adapter"]
    primal2_path = Path(primal2["path"])
    if not primal2_path.is_file() or sha256(primal2_path) != primal2["sha256"]:
        raise RuntimeError("frozen artifact mismatch: primal2_adapter")
    for key in ("certified_manifest", "lifelong_manifest"):
        path = ROOT / manifest["inputs"][key]
        expected = manifest["inputs"][f"{key}_sha256"]
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen input mismatch: {key}")


def commands() -> list[list[tuple[str, list[str]]]]:
    python = sys.executable
    oneshot = str(ROOT / "tools/run_final_certified_oneshot.py")
    stochastic = str(ROOT / "tools/run_final_stochastic.py")
    freeze = str(ARTIFACT_MANIFEST)
    certified = str(ROOT / "results/revision_final/certified_inputs_v3/MANIFEST.json")
    lifelong_inputs = str(ROOT / "results/revision_final/lifelong_inputs_v2/MANIFEST.json")
    lima = str(ROOT / "results/revision_final/frozen_artifacts_step_v4_optimized/lima")
    primal_python = str(Path.home() / "miniconda3/envs/primal2/bin/python")
    primal_script = str(Path.home() / "mapf-baselines/PRIMAL2-opt/run_our_instances_live.py")
    primal_model = str(Path.home() / "mapf-baselines/PRIMAL2/model_primal2_oneshot")
    return [
        [
            ("oneshot_lima", [
                python, oneshot, "--algorithm", "lima", "--jobs", "12",
                "--input-manifest", certified, "--freeze-manifest", freeze,
                "--lima-binary", lima,
                "--output-dir", str(ROOT / "results/revision_final/oneshot_lima_certified_step_v4_optimized"),
            ]),
            ("oneshot_primal2", [
                python, oneshot, "--algorithm", "primal2", "--jobs", "5",
                "--input-manifest", certified, "--freeze-manifest", freeze,
                "--lima-binary", lima,
                "--primal-python", primal_python,
                "--primal-script", primal_script,
                "--primal-model", primal_model,
                "--max-steps", "5000",
                "--primal-stall-steps", "256",
                "--reuse-records-from", str(ROOT / "results/revision_final/oneshot_primal2_certified_step_v4_optimized_r1"),
                "--reuse-records-from", str(ROOT / "results/revision_final/oneshot_primal2_prefetch_wh10_v4r1"),
                "--reuse-records-from", str(ROOT / "results/revision_final/oneshot_primal2_prefetch_wh20_v6_common5000"),
                "--output-dir", str(ROOT / "results/revision_final/oneshot_primal2_certified_step_v6_common5000_stall256"),
            ]),
        ],
        [
            ("stochastic_lima", [
                python, stochastic, "--algorithm", "lima", "--jobs", "10",
                "--input-manifest", certified, "--lima", lima,
                "--max-steps", "5000", "--densities", "10,20,30",
                "--probabilities", "0.05,0.10,0.15,0.20",
                "--output-dir", str(ROOT / "results/revision_final/stochastic_lima_step_v7_pgrid5_d10_30"),
            ]),
            ("stochastic_pibt", [
                python, stochastic, "--algorithm", "pibt", "--jobs", "5",
                "--input-manifest", certified, "--lima", lima,
                "--max-steps", "5000", "--densities", "10,20,30",
                "--probabilities", "0.05,0.10,0.15,0.20",
                "--pibt-binary", str(ROOT / "results/revision_final/frozen_artifacts_step_v2/pibt"),
                "--output-dir", str(ROOT / "results/revision_final/stochastic_pibt_replan_step_v1_pgrid5_d10_30"),
            ]),
            ("stochastic_lacam_replan", [
                python, stochastic, "--algorithm", "lacam-replan", "--jobs", "5",
                "--input-manifest", certified, "--lima", lima,
                "--max-steps", "5000", "--densities", "10,20,30",
                "--probabilities", "0.05,0.10,0.15,0.20",
                "--lacam-binary", str(ROOT / "results/revision_final/frozen_artifacts_step_v2/lacam"),
                "--lacam-max-iterations", "100000",
                "--output-dir", str(ROOT / "results/revision_final/stochastic_lacam_replan_step_v1_pgrid5_d10_30"),
            ]),
        ],
        [
            ("lifelong", [
                python, str(ROOT / "tools/run_final_lifelong.py"), "--jobs", "12",
                "--input-manifest", lifelong_inputs, "--binary", lima,
                "--output-dir", str(ROOT / "results/revision_final/lifelong_lima_step_v4_optimized"),
            ]),
            ("admission", [
                python, str(ROOT / "tools/run_final_admission_ablation.py"), "--jobs", "6",
                "--freeze-manifest", freeze, "--certified-manifest", certified,
                "--binary", lima,
                "--output-dir", str(ROOT / "results/revision_final/admission_ablation_step_v4_optimized"),
            ]),
            ("local_solver", [
                python, str(ROOT / "tools/run_final_local_solver.py"),
                "--freeze-manifest", freeze,
                "--output-dir", str(ROOT / "results/revision_final/local_solver_reference_step_v4_optimized"),
            ]),
        ],
    ]


def run_phase(index: int, phase: list[tuple[str, list[str]]], log_dir: Path) -> dict[str, int]:
    processes: dict[str, tuple[subprocess.Popen, object]] = {}
    for name, command in phase:
        log_path = log_dir / f"{name}.log"
        stream = log_path.open("a", encoding="utf-8")
        stream.write(
            f"\n[{datetime.now(timezone.utc).isoformat()}] phase={index} "
            f"command={json.dumps(command)}\n"
        )
        stream.flush()
        process = subprocess.Popen(
            command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
            text=True, start_new_session=True,
        )
        processes[name] = (process, stream)
        print(f"phase {index}: started {name} pid={process.pid}", flush=True)
    returncodes: dict[str, int] = {}
    for name, (process, stream) in processes.items():
        returncode = process.wait()
        stream.write(
            f"[{datetime.now(timezone.utc).isoformat()}] returncode={returncode}\n")
        stream.close()
        returncodes[name] = returncode
        print(f"phase {index}: finished {name} rc={returncode}", flush=True)
    return returncodes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-root", default=str(QUEUE_ROOT))
    args = parser.parse_args()
    queue_root = Path(args.queue_root).resolve()
    queue_root.mkdir(parents=True, exist_ok=True)
    log_dir = queue_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    artifact_manifest = json.loads(ARTIFACT_MANIFEST.read_text(encoding="utf-8"))
    verify_frozen_artifacts(artifact_manifest)
    phases = commands()
    queue_spec = {
        "schema_version": 1,
        "artifact_manifest": str(ARTIFACT_MANIFEST.relative_to(ROOT)),
        "artifact_manifest_sha256": sha256(ARTIFACT_MANIFEST),
        "protocol_commit": artifact_manifest["protocol_commit"],
        "phases": [
            [{"name": name, "command": command} for name, command in phase]
            for phase in phases
        ],
        "termination_policy": "step/search limits only; no wall-clock cutoff",
    }
    atomic_json(queue_root / "MANIFEST.json", queue_spec)

    lock = queue_root / ".RUNNING"
    if lock.is_file():
        try:
            old_pid = int(json.loads(lock.read_text(encoding="utf-8"))["pid"])
            os.kill(old_pid, 0)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass
        else:
            parser.error(f"queue is already running with pid {old_pid}")
    atomic_json(lock, {
        "pid": os.getpid(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    })

    completed: dict[str, int] = {}
    try:
        for index, phase in enumerate(phases, 1):
            returncodes = run_phase(index, phase, log_dir)
            completed.update(returncodes)
            atomic_json(queue_root / "STATUS.json", {
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "completed": completed,
                "last_phase": index,
            })
            failures = {name: code for name, code in returncodes.items() if code != 0}
            if failures:
                print(f"phase {index} infrastructure failures: {failures}", flush=True)
                return 1
    finally:
        lock.unlink(missing_ok=True)
    atomic_json(queue_root / "COMPLETE.json", {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "returncodes": completed,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
