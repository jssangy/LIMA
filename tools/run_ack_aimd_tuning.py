#!/usr/bin/env python3
"""Tune acknowledged AIMD on the disjoint Gate-C development instances."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = ((0.25, 0.25), (0.25, 0.50), (0.50, 0.25),
              (0.50, 0.50), (0.75, 0.25), (0.75, 0.50))


@dataclass(frozen=True)
class Cell:
    name: str
    map_file: str
    scenario_file: str
    agents: int


CELLS = (
    Cell("warehouse_10_20_d50_s0", "data/maps/warehouse_10_20_paper.map",
         "data/scenarios/warehouse-10-20-paper/warehouse-10-20-paper_s0.scen", 1324),
    Cell("warehouse_20_40_d30_s0", "data/maps/warehouse_20_40_paper.map",
         "data/scenarios/warehouse-20-40-paper/warehouse-20-40-paper_s0.scen", 3149),
    Cell("cross_3030_d50_s0", "data/maps/cross_3030_paper.map",
         "data/scenarios/cross-30-30-paper/cross-30-30-paper_s0.scen", 5100),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False,
                                     encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def parse_summary(text: str) -> dict[str, str]:
    lines = [line for line in text.splitlines() if line.strip()]
    return dict(re.findall(r"(\w+)=([^\s]+)", lines[-1])) if lines else {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="build_delay_review/lima")
    parser.add_argument("--output-dir", default="results/ack_aimd_tuning_v1")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=50_000)
    args = parser.parse_args()
    if args.jobs < 1 or args.max_steps < 1:
        parser.error("--jobs and --max-steps must be positive")

    binary = (ROOT / args.binary).resolve()
    output = (ROOT / args.output_dir).resolve()
    records = output / "records"
    records.mkdir(parents=True, exist_ok=True)
    jobs = [(cell, beta, alpha, probability)
            for cell in CELLS for beta, alpha in PARAMETERS
            for probability in (0.0, 0.20)]

    def run(spec: tuple[Cell, float, float, float]) -> dict:
        cell, beta, alpha, probability = spec
        tag = f"{cell.name}_b{int(beta*100):02d}_a{int(alpha*100):02d}_p{int(probability*100):02d}"
        record_file = records / f"{tag}.json"
        if record_file.exists():
            return json.loads(record_file.read_text(encoding="utf-8"))
        command = [
            str(binary), "--profile", "lima-default", "--mode", "solve",
            "--map", cell.map_file, "--scenario", cell.scenario_file,
            "--agents", str(cell.agents), "--seed", "0",
            "--max-steps", str(args.max_steps),
            "--stall-threshold", str(args.max_steps + 1),
            "--failure-prob", str(probability), "--goal-behavior", "disappear",
            "--gate-policy", "aimd", "--gate-param", str(beta),
            "--gate-param2", str(alpha), "--no-trace",
        ]
        process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        result = parse_summary(process.stdout)
        payload = {
            "tag": tag, "cell": cell.name, "agents": cell.agents,
            "beta": beta, "alpha": alpha, "probability": probability,
            "returncode": process.returncode, "result": result,
            "stderr": process.stderr.strip(), "command": command,
        }
        write_json(record_file, payload)
        print(f"{tag} {result.get('status')} {result.get('steps')} "
              f"{result.get('completed')}", flush=True)
        return payload

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        rows = list(pool.map(run, jobs))
    rows.sort(key=lambda row: (row["beta"], row["alpha"], row["probability"], row["cell"]))
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("beta", "alpha", "probability", "cell", "status",
                         "completed", "steps", "moves", "waits", "returncode"))
        for row in rows:
            result = row["result"]
            writer.writerow((row["beta"], row["alpha"], row["probability"], row["cell"],
                             result.get("status"), result.get("completed"), result.get("steps"),
                             result.get("moves"), result.get("waits"), row["returncode"]))
    write_json(output / "MANIFEST.json", {
        "binary": str(binary), "binary_sha256": sha256(binary),
        "max_steps": args.max_steps, "jobs": len(jobs),
        "development_instances": [cell.__dict__ for cell in CELLS],
        "parameters": PARAMETERS, "probabilities": (0.0, 0.20),
    })
    return 0 if all(row["returncode"] == 0 for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
