#!/usr/bin/env python3
"""Backfill only early-stopped stochastic LaCAM and PIBT cells."""

from __future__ import annotations

import concurrent.futures
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
REVISION = ROOT / "results/revision_final"
QUEUE = REVISION / "stochastic_early_stop_backfill_queue_v1"
SOURCE = {
    "lacam-replan": REVISION / "stochastic_lacam_replan_core_s0_4_v1",
    "pibt": REVISION / "stochastic_pibt_core_s0_4_v1",
}
EXPECTED_SOURCE_RECORDS = 180


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_files(campaign: Path) -> list[Path]:
    return sorted((campaign / "records").glob("*.json"))


def lock_pid(campaign: Path) -> int | None:
    lock = campaign / ".RUNNING"
    if not lock.is_file():
        return None
    try:
        first_line = lock.read_text(encoding="utf-8").splitlines()[0]
        return int(first_line.split("=", 1)[1])
    except (IndexError, ValueError):
        return -1


def pid_alive(pid: int | None) -> bool:
    return pid is not None and pid > 0 and Path(f"/proc/{pid}").exists()


def wait_for_sources() -> None:
    while True:
        ready = True
        report = []
        for algorithm, campaign in SOURCE.items():
            count = len(record_files(campaign))
            pid = lock_pid(campaign)
            alive = pid_alive(pid)
            report.append(f"{algorithm}={count}/{EXPECTED_SOURCE_RECORDS},alive={alive}")
            if count != EXPECTED_SOURCE_RECORDS or alive:
                ready = False
            if not alive and count < EXPECTED_SOURCE_RECORDS:
                raise RuntimeError(
                    f"source stopped early: {algorithm} {count}/{EXPECTED_SOURCE_RECORDS}")
        print(f"[{utc_now()}] " + "; ".join(report), flush=True)
        if ready:
            return
        time.sleep(60)


def early_stop_groups(algorithm: str, campaign: Path) -> list[tuple[str, int, float]]:
    groups: set[tuple[str, int, float]] = set()
    for path in record_files(campaign):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") == "early_stopped_after_zero_success":
            groups.add((record["map"], int(record["density"]), float(record["probability"])))
    return sorted(groups)


def probability_tag(probability: float) -> str:
    return f"p{int(round(100 * probability)):02d}"


def run_group(job: tuple[str, str, int, float]) -> dict:
    algorithm, map_name, density, probability = job
    group_tag = f"{map_name}_d{density:02d}_{probability_tag(probability)}"
    output = (
        f"results/revision_final/stochastic_{algorithm.replace('-', '_')}"
        f"_early_stop_backfill_v1/groups/{group_tag}"
    )
    trace_root = (
        f"results/revision_final/stochastic_trace_descriptors_"
        f"{algorithm.replace('-', '_')}_backfill_v1"
    )
    command = [
        "python3", "tools/run_final_stochastic.py",
        "--algorithm", algorithm,
        "--input-manifest", "results/revision_final/certified_inputs_v3/MANIFEST.json",
        "--maps", map_name,
        "--densities", str(density),
        "--scenarios", "0-4",
        "--probabilities", str(probability),
        "--max-steps", "5000",
        "--jobs", "3",
        "--no-early-stop",
        "--lima", "results/revision_final/frozen_artifacts_step_v4_optimized/lima",
        "--pibt-binary", "results/revision_final/frozen_artifacts_step_v2/pibt",
        "--lacam-binary", "results/revision_final/frozen_artifacts_step_v2/lacam",
        "--lacam-max-iterations", "100000",
        "--output-dir", output,
        "--trace-root", trace_root,
    ]
    QUEUE.mkdir(parents=True, exist_ok=True)
    log_path = QUEUE / f"{algorithm.replace('-', '_')}_{group_tag}.log"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{utc_now()}] {' '.join(command)}\n")
        log.flush()
        process = subprocess.run(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            text=True, check=False,
        )
    if process.returncode != 0:
        raise RuntimeError(f"backfill failed ({process.returncode}): {algorithm}/{group_tag}")

    output_path = ROOT / output
    records = [json.loads(path.read_text(encoding="utf-8"))
               for path in record_files(output_path)]
    if len(records) != 5:
        raise RuntimeError(f"backfill record mismatch: {algorithm}/{group_tag} {len(records)}/5")
    for record in records:
        if record.get("returncode") not in (0, 2):
            raise RuntimeError(f"unexpected returncode: {algorithm}/{record['tag']}")
        result = record.get("result", {})
        if "completed" not in result or "steps" not in result:
            raise RuntimeError(f"missing agent completion: {algorithm}/{record['tag']}")
        if result.get("vertex_conflicts") != "0" or result.get("edge_conflicts") != "0":
            raise RuntimeError(f"path conflict: {algorithm}/{record['tag']}")
        if "communication_events" not in result:
            raise RuntimeError(f"missing communication telemetry: {algorithm}/{record['tag']}")
    return {
        "algorithm": algorithm,
        "map": map_name,
        "density": density,
        "probability": probability,
        "records": len(records),
        "output_dir": output,
        "returncode": process.returncode,
    }


def main() -> int:
    QUEUE.mkdir(parents=True, exist_ok=True)
    wait_for_sources()
    jobs = [
        (algorithm, map_name, density, probability)
        for algorithm, campaign in SOURCE.items()
        for map_name, density, probability in early_stop_groups(algorithm, campaign)
    ]
    print(f"[{utc_now()}] groups={len(jobs)} records={5 * len(jobs)}", flush=True)
    completed: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_group, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            completed.append(result)
            print(
                f"[{len(completed)}/{len(jobs)}] {result['algorithm']} "
                f"{result['map']} d{result['density']:02d} p={result['probability']}",
                flush=True,
            )
    status = {
        "schema_version": 1,
        "completed_utc": utc_now(),
        "source_campaigns": {key: str(value.relative_to(ROOT)) for key, value in SOURCE.items()},
        "group_count": len(jobs),
        "record_count": sum(item["records"] for item in completed),
        "groups": sorted(completed, key=lambda item: (
            item["algorithm"], item["map"], item["probability"], item["density"])),
    }
    (QUEUE / "STATUS.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
