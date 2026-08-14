#!/usr/bin/env python3
"""Build the Gate C/D tournament tables from the .line result files.

Usage: tools/gating_report.py <dir> [<dir>...]
Emits a markdown variant x cell matrix (completion) plus a deterministic-field
table, and lists cells where a variant regressed against its own baseline.
"""
import re
import sys
from pathlib import Path

FIELDS = ("status", "steps", "completed", "moves", "waits", "deadlocks")
TIMEOUT = 2400  # per-cell wall budget used by every harness in this study


def parse(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    parts = text.split("|")
    if len(parts) < 5:
        return None
    mp, agents, seed, variant, rest = parts[0], parts[1], parts[2], parts[3], "|".join(parts[4:])
    d = dict(re.findall(r"(\w+)=([^\s]+)", rest))
    if "completed" not in d:
        # Empty tail: the run was killed by the per-cell wall clock, which is a
        # result (a DNF), not a missing measurement.
        return {
            "cell": f"{mp} a{agents} s{seed}",
            "map": mp, "agents": int(agents), "seed": int(seed), "variant": variant,
            "status": "wall_timeout", "steps": 0, "done": 0, "total": int(agents),
            "moves": 0, "waits": 0, "deadlocks": 0, "elapsed": float(TIMEOUT),
        }
    done, total = (int(x) for x in d["completed"].split("/"))
    return {
        "cell": f"{mp} a{agents} s{seed}",
        "map": mp, "agents": int(agents), "seed": int(seed), "variant": variant,
        "status": d.get("status", "?"), "steps": int(d.get("steps", 0)),
        "done": done, "total": total, "moves": int(d.get("moves", 0)),
        "waits": int(d.get("waits", 0)), "deadlocks": int(d.get("deadlocks", 0)),
        "elapsed": float(d.get("elapsed_seconds", 0.0)),
    }


def main() -> None:
    rows = []
    for d in sys.argv[1:]:
        for f in sorted(Path(d).glob("*.line")):
            r = parse(f)
            if r:
                rows.append(r)
    cells = sorted({(r["map"], r["agents"], r["seed"]) for r in rows})
    variants = sorted({r["variant"] for r in rows})
    by = {(r["cell"], r["variant"]): r for r in rows}

    print("| cell | " + " | ".join(variants) + " |")
    print("|---|" + "---|" * len(variants))
    for mp, a, s in cells:
        cell = f"{mp} a{a} s{s}"
        out = []
        for v in variants:
            r = by.get((cell, v))
            if not r:
                out.append("-")
            elif r["status"] == "completed":
                out.append(f"OK {r['steps']}")
            elif r["status"] == "wall_timeout":
                out.append("**TIMEOUT**")
            else:
                out.append(f"**{r['done']}/{r['total']}**")
        print(f"| {cell} | " + " | ".join(out) + " |")

    print()
    print("Per-variant completion count (cells run / cells complete):")
    for v in variants:
        rs = [r for r in rows if r["variant"] == v]
        ok = sum(1 for r in rs if r["status"] == "completed")
        print(f"  {v:24s} {ok:3d}/{len(rs):3d}")


if __name__ == "__main__":
    main()
