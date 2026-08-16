#!/usr/bin/env python3
"""Finish the paper-facing LIMA experiments in priority order.

The queue adopts already-running campaigns, resumes only when their runner is
gone, and postpones ultra-high-density LIMA and medium-or-higher PRIMAL2 work
until the paper-critical one-shot telemetry, stochastic, and lifelong results
are complete.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results/revision_final"
PYTHON = sys.executable


@dataclass(frozen=True)
class Campaign:
    name: str
    expected: int
    command: tuple[str, ...] | None = None

    @property
    def root(self) -> Path:
        return RESULTS / self.name


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8"
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def record_paths(campaign: Campaign) -> list[Path]:
    return sorted((campaign.root / "records").glob("*.json"))


def running_for(campaign: Campaign) -> bool:
    tokens = {str(campaign.root), str(campaign.root.relative_to(ROOT))}
    for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            text = cmdline.read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
        except (OSError, PermissionError):
            continue
        if any(token in text for token in tokens):
            return True
    return False


def audit(campaign: Campaign) -> dict:
    malformed = 0
    unexpected = 0
    validation_errors = 0
    for path in record_paths(campaign):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            malformed += 1
            continue
        if record.get("returncode") not in (0, None, 2):
            unexpected += 1
        if record.get("timed_out") is True:
            unexpected += 1
        if record.get("validation_error") or record.get("capacity_violation"):
            validation_errors += 1
        equivalence = record.get("scalar_equivalence") or {}
        if equivalence and equivalence.get("all_match") is not True:
            validation_errors += 1
        telemetry = record.get("telemetry") or {}
        trajectory = record.get("trajectory") or {}
        result = record.get("result") or {}
        for conformity in (
            telemetry.get("path_conformity") or {},
            trajectory.get("path_conformity") or {},
        ):
            if conformity.get("online_validation_ok") is False:
                validation_errors += 1
        for source in (telemetry, trajectory, result):
            for key in ("vertex_conflicts", "edge_conflicts", "invalid_moves"):
                if source.get(key) not in (None, 0, "0"):
                    validation_errors += 1
    records = len(record_paths(campaign))
    running = running_for(campaign)
    locked = (campaign.root / ".RUNNING").exists()
    return {
        "records": records,
        "expected": campaign.expected,
        "running": running,
        "locked": locked,
        "malformed": malformed,
        "unexpected": unexpected,
        "validation_errors": validation_errors,
        "ready": (
            records == campaign.expected
            and not running
            and malformed == 0
            and unexpected == 0
            and validation_errors == 0
        ),
    }


def start(campaign: Campaign, queue_root: Path) -> None:
    if campaign.command is None:
        raise RuntimeError(f"required fixed campaign is incomplete: {campaign.name}")
    lock = campaign.root / ".RUNNING"
    if lock.exists() and not running_for(campaign):
        lock.unlink()
    campaign.root.mkdir(parents=True, exist_ok=True)
    log = queue_root / "logs" / f"{campaign.name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(
            f"\n[{datetime.now(timezone.utc).isoformat()}] "
            f"command={json.dumps(campaign.command)}\n"
        )
        stream.flush()
        subprocess.Popen(
            campaign.command,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )


def wait_phase(
    name: str,
    campaigns: tuple[Campaign, ...],
    queue_root: Path,
    poll_seconds: int,
) -> None:
    starts: dict[str, int] = {campaign.name: 0 for campaign in campaigns}
    while True:
        rows = {campaign.name: audit(campaign) for campaign in campaigns}
        atomic_json(queue_root / f"AUDIT_{name}.json", rows)
        if all(row["ready"] for row in rows.values()):
            return
        for campaign in campaigns:
            row = rows[campaign.name]
            if row["ready"] or row["running"]:
                continue
            if row["records"] > campaign.expected:
                raise RuntimeError(
                    f"record count exceeds expectation: {campaign.name}"
                )
            if row["malformed"] or row["unexpected"] or row["validation_errors"]:
                raise RuntimeError(f"audit failure: {campaign.name}: {row}")
            if starts[campaign.name] >= 3:
                raise RuntimeError(f"campaign repeatedly stopped: {campaign.name}")
            start(campaign, queue_root)
            starts[campaign.name] += 1
        print(
            f"[{datetime.now(timezone.utc).isoformat()}] phase={name} "
            + ", ".join(
                f"{campaign.name}={rows[campaign.name]['records']}/{campaign.expected}"
                for campaign in campaigns
            ),
            flush=True,
        )
        time.sleep(poll_seconds)


def run_derived(name: str, command: tuple[str, ...], queue_root: Path) -> None:
    log = queue_root / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(
            f"\n[{datetime.now(timezone.utc).isoformat()}] "
            f"command={json.dumps(command)}\n"
        )
        stream.flush()
        result = subprocess.run(
            command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
            text=True, check=False,
        )
        stream.write(f"returncode={result.returncode}\n")
    if result.returncode != 0:
        raise RuntimeError(f"derived command failed: {name}")


def commands() -> tuple[
    tuple[Campaign, ...],
    tuple[Campaign, ...],
    tuple[Campaign, ...],
    tuple[Campaign, ...],
]:
    certified = "results/revision_final/certified_inputs_v3/MANIFEST.json"
    lifelong_inputs = "results/revision_final/lifelong_inputs_v2/MANIFEST.json"
    freeze = "results/revision_final/frozen_artifacts_step_v4_optimized_r2/MANIFEST.json"
    lima = "results/revision_final/frozen_artifacts_step_v4_optimized/lima"
    pibt = "results/revision_final/frozen_artifacts_step_v2/pibt"
    lacam = "results/revision_final/frozen_artifacts_step_v2/lacam"
    primal_python = str(Path.home() / "miniconda3/envs/primal2/bin/python3.7")
    primal_script = str(Path.home() / "mapf-baselines/PRIMAL2-opt/run_our_instances_live.py")
    primal_model = str(Path.home() / "mapf-baselines/PRIMAL2/model_primal2_oneshot")

    core = (
        Campaign("oneshot_lima_certified_step_v4_optimized", 274),
        Campaign("oneshot_cbs_certified_step_v3", 280),
        Campaign("oneshot_lacam_certified_step_v3", 280),
        Campaign("oneshot_pibt_certified_step_v3", 280),
        Campaign("oneshot_cbs_telemetry_backfill_v1", 280),
        Campaign("oneshot_lacam_telemetry_backfill_v2", 280),
        Campaign(
            "oneshot_pibt_telemetry_backfill_v2", 280,
            (PYTHON, "tools/backfill_oneshot_baseline_telemetry.py",
             "--algorithm", "pibt", "--campaign",
             "results/revision_final/oneshot_pibt_certified_step_v3",
             "--output-dir",
             "results/revision_final/oneshot_pibt_telemetry_backfill_v2",
             "--jobs", "3"),
        ),
        Campaign("local_solver_core_v1", 9),
        Campaign(
            "stochastic_lima_core_s0_4_v1", 180,
            (PYTHON, "tools/run_final_stochastic.py", "--algorithm", "lima",
             "--jobs", "10", "--input-manifest", certified,
             "--densities", "10,20,30", "--scenarios", "0-4",
             "--probabilities", "0.05,0.10,0.15,0.20",
             "--max-steps", "5000", "--lima", lima, "--output-dir",
             "results/revision_final/stochastic_lima_core_s0_4_v1"),
        ),
        Campaign(
            "stochastic_pibt_core_s0_4_v1", 180,
            (PYTHON, "tools/run_final_stochastic.py", "--algorithm", "pibt",
             "--jobs", "3", "--input-manifest", certified,
             "--densities", "10,20,30", "--scenarios", "0-4",
             "--probabilities", "0.05,0.10,0.15,0.20",
             "--max-steps", "5000", "--lima", lima,
             "--pibt-binary", pibt, "--output-dir",
             "results/revision_final/stochastic_pibt_core_s0_4_v1"),
        ),
        Campaign(
            "stochastic_lacam_replan_core_s0_4_v1", 180,
            (PYTHON, "tools/run_final_stochastic.py", "--algorithm", "lacam-replan",
             "--jobs", "3", "--input-manifest", certified,
             "--densities", "10,20,30", "--scenarios", "0-4",
             "--probabilities", "0.05,0.10,0.15,0.20",
             "--max-steps", "5000", "--lima", lima,
             "--lacam-binary", lacam, "--lacam-max-iterations", "100000",
             "--output-dir",
             "results/revision_final/stochastic_lacam_replan_core_s0_4_v1"),
        ),
        Campaign(
            "lifelong_lima_core_s0_4_v1", 90,
            (PYTHON, "tools/run_final_lifelong.py", "--input-manifest",
             lifelong_inputs, "--binary", lima, "--densities", "10,30,50",
             "--scenarios", "0,1,2,3,4", "--horizon", "10000",
             "--warmup", "1000", "--jobs", "4", "--output-dir",
             "results/revision_final/lifelong_lima_core_s0_4_v1"),
        ),
    )

    confirmatory = (
        Campaign(
            "stochastic_lima_confirm_s5_9_v1", 180,
            (PYTHON, "tools/run_final_stochastic.py", "--algorithm", "lima",
             "--jobs", "10", "--input-manifest", certified,
             "--densities", "10,20,30", "--scenarios", "5,6,7,8,9",
             "--probabilities", "0.05,0.10,0.15,0.20",
             "--max-steps", "5000", "--lima", lima, "--output-dir",
             "results/revision_final/stochastic_lima_confirm_s5_9_v1"),
        ),
        Campaign(
            "stochastic_pibt_confirm_s5_9_v1", 180,
            (PYTHON, "tools/run_final_stochastic.py", "--algorithm", "pibt",
             "--jobs", "3", "--input-manifest", certified,
             "--densities", "10,20,30", "--scenarios", "5,6,7,8,9",
             "--probabilities", "0.05,0.10,0.15,0.20",
             "--max-steps", "5000", "--lima", lima,
             "--pibt-binary", pibt, "--output-dir",
             "results/revision_final/stochastic_pibt_confirm_s5_9_v1"),
        ),
        Campaign(
            "stochastic_lacam_replan_confirm_s5_9_v1", 180,
            (PYTHON, "tools/run_final_stochastic.py", "--algorithm", "lacam-replan",
             "--jobs", "3", "--input-manifest", certified,
             "--densities", "10,20,30", "--scenarios", "5,6,7,8,9",
             "--probabilities", "0.05,0.10,0.15,0.20",
             "--max-steps", "5000", "--lima", lima,
             "--lacam-binary", lacam, "--lacam-max-iterations", "100000",
             "--output-dir",
             "results/revision_final/stochastic_lacam_replan_confirm_s5_9_v1"),
        ),
        Campaign(
            "lifelong_lima_confirm_s5_9_v1", 90,
            (PYTHON, "tools/run_final_lifelong.py", "--input-manifest",
             lifelong_inputs, "--binary", lima, "--densities", "10,30,50",
             "--scenarios", "5,6,7,8,9", "--horizon", "10000",
             "--warmup", "1000", "--jobs", "4", "--output-dir",
             "results/revision_final/lifelong_lima_confirm_s5_9_v1"),
        ),
    )

    bonus_and_low = (
        Campaign(
            "bonus_mapf_lns2_common5000_priority_v1", 280,
            (PYTHON, "tools/run_bonus_mapf_lns2.py", "--jobs", "4",
             "--max-steps", "5000", "--max-iterations", "100000",
             "--output-dir",
             "results/revision_final/bonus_mapf_lns2_common5000_priority_v1"),
        ),
        Campaign(
            "bonus_lifelong_pibt_priority_v1", 90,
            (PYTHON, "tools/run_bonus_lifelong_pibt.py", "--jobs", "4",
             "--densities", "10,30,50", "--scenarios", "0,1,2,3,4,5,6,7,8,9",
             "--horizon", "10000", "--warmup", "1000", "--output-dir",
             "results/revision_final/bonus_lifelong_pibt_priority_v1"),
        ),
        Campaign(
            "bonus_pibt_agent_completion_priority_v1", 280,
            (PYTHON, "tools/run_final_certified_oneshot.py", "--algorithm", "pibt",
             "--jobs", "4", "--no-early-stop", "--max-steps", "5000",
             "--freeze-manifest", freeze, "--lima-binary", lima,
             "--pibt-binary",
             str(Path.home() / "mapf-baselines/pibt2/build_bonus/mapf"),
             "--output-dir",
             "results/revision_final/bonus_pibt_agent_completion_priority_v1"),
        ),
        Campaign(
            "oneshot_primal2_low_deferred_v1", 60,
            (PYTHON, "tools/run_final_certified_oneshot.py", "--algorithm", "primal2",
             "--jobs", "2", "--targets", "d01,d05", "--input-manifest",
             certified, "--freeze-manifest", freeze, "--lima-binary", lima,
             "--primal-python", primal_python, "--primal-script", primal_script,
             "--primal-model", primal_model, "--max-steps", "5000",
             "--primal-stall-steps", "256", "--reuse-records-from",
             "results/revision_final/oneshot_primal2_certified_step_v6_common5000_stall256",
             "--reuse-records-from",
             "results/revision_final/oneshot_primal2_certified_step_v4_optimized_r1",
             "--reuse-records-from",
             "results/revision_final/oneshot_primal2_prefetch_wh10_v4r1",
             "--reuse-records-from",
             "results/revision_final/oneshot_primal2_prefetch_wh20_v6_common5000",
             "--output-dir",
             "results/revision_final/oneshot_primal2_low_deferred_v1"),
        ),
    )

    deferred = (
        Campaign(
            "admission_ablation_core_v1", 24,
            (PYTHON, "tools/run_final_admission_ablation.py", "--jobs", "6",
             "--freeze-manifest", freeze, "--certified-manifest", certified,
             "--binary", lima, "--output-dir",
             "results/revision_final/admission_ablation_core_v1"),
        ),
        Campaign(
            "oneshot_lima_boundary_deferred_v1", 6,
            (PYTHON, "tools/run_final_certified_oneshot.py", "--algorithm", "lima",
             "--jobs", "6", "--maps", "warehouse_20_40", "--targets", "boundary",
             "--scenarios", "0,2,4,5,8,9", "--max-steps", "100000",
             "--input-manifest", certified, "--freeze-manifest", freeze,
             "--lima-binary", lima, "--output-dir",
             "results/revision_final/oneshot_lima_boundary_deferred_v1"),
        ),
        Campaign(
            "oneshot_primal2_midplus_deferred_v1", 220,
            (PYTHON, "tools/run_final_certified_oneshot.py", "--algorithm", "primal2",
             "--jobs", "2", "--targets",
             "d10,d20,d30,d40,d50,d60,d65,d70,boundary",
             "--input-manifest", certified, "--freeze-manifest", freeze,
             "--lima-binary", lima, "--primal-python", primal_python,
             "--primal-script", primal_script, "--primal-model", primal_model,
             "--max-steps", "5000", "--primal-stall-steps", "256",
             "--reuse-records-from",
             "results/revision_final/oneshot_primal2_certified_step_v6_common5000_stall256",
             "--reuse-records-from",
             "results/revision_final/oneshot_primal2_certified_step_v4_optimized_r1",
             "--reuse-records-from",
             "results/revision_final/oneshot_primal2_low_deferred_v1",
             "--output-dir",
             "results/revision_final/oneshot_primal2_midplus_deferred_v1"),
        ),
    )
    return core, confirmatory, bonus_and_low, deferred


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--queue-root",
        default="results/revision_final/priority_continuation_v1",
    )
    parser.add_argument("--poll-seconds", type=int, default=300)
    args = parser.parse_args()
    if args.poll_seconds < 60:
        parser.error("--poll-seconds must be at least 60")
    queue_root = (ROOT / args.queue_root).resolve()
    queue_root.mkdir(parents=True, exist_ok=True)
    lock = queue_root / ".RUNNING"
    if lock.exists():
        parser.error(f"priority queue lock exists: {lock}")
    atomic_json(lock, {
        "pid": os.getpid(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    })
    core, confirmatory, bonus_and_low, deferred = commands()
    atomic_json(queue_root / "MANIFEST.json", {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "ordering": [
            "paper-critical core",
            "core summaries",
            "confirmatory s5-s9",
            "bonus and remaining low-density PRIMAL2",
            "deferred ultra-high-density LIMA/admission and PRIMAL2 d10+",
            "final summaries",
        ],
        "termination": "step/search limits only; no wall-clock cutoff",
        "phases": {
            "core": {c.name: c.expected for c in core},
            "confirmatory": {c.name: c.expected for c in confirmatory},
            "bonus_and_low": {c.name: c.expected for c in bonus_and_low},
            "deferred": {c.name: c.expected for c in deferred},
        },
    })
    try:
        wait_phase("core", core, queue_root, args.poll_seconds)
        run_derived(
            "paper_metrics_core",
            (PYTHON, "tools/summarize_paper_metrics.py", "--output-dir",
             "results/revision_final/paper_metrics_core_v1"),
            queue_root,
        )
        run_derived(
            "lifelong_core_summary",
            (PYTHON, "tools/summarize_lifelong.py",
             str(RESULTS / "lifelong_lima_core_s0_4_v1")),
            queue_root,
        )
        wait_phase("confirmatory", confirmatory, queue_root, args.poll_seconds)
        wait_phase("bonus_and_low", bonus_and_low, queue_root, args.poll_seconds)
        wait_phase("deferred", deferred, queue_root, args.poll_seconds)
        run_derived(
            "paper_metrics_final",
            (PYTHON, "tools/summarize_paper_metrics.py",
             "--campaign",
             "mapf-lns2=results/revision_final/bonus_mapf_lns2_common5000_priority_v1",
             "--campaign",
             "pibt-acr=results/revision_final/bonus_pibt_agent_completion_priority_v1",
             "--output-dir", "results/revision_final/paper_metrics_priority_final_v1"),
            queue_root,
        )
        atomic_json(queue_root / "COMPLETE.json", {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
        })
    finally:
        lock.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
