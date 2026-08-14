#!/usr/bin/env python3
"""Report provenance and geometry of every map file, current and historical."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = {  # Table 2 of the submitted manuscript
    "Standard 1": (161, 63, 2649, 189, 390),
    "Standard 2": (321, 123, 10499, 779, 1580),
    "Square 1": (267, 187, 10200, 900, 1860),
}


def sh(cmd):
    out = subprocess.run(cmd, cwd=ROOT, shell=True, capture_output=True, text=True)
    return out.stdout.strip()


def geometry(text):
    lines = text.splitlines()
    body = []
    height = width = None
    for line in lines:
        low = line.strip().lower()
        if low.startswith("height"):
            height = int(low.split()[1])
        elif low.startswith("width"):
            width = int(low.split()[1])
        elif low.startswith(("type", "map")):
            continue
        elif line.strip():
            body.append(line.rstrip("\n"))
    if body and height is None:
        height, width = len(body), max(len(r) for r in body)
    free = sum(1 for row in body for ch in row if ch in ".GS")
    sinks = sum(1 for row in body for ch in row if ch == "S")
    blocked = sum(1 for row in body for ch in row if ch in "@T")
    return width, height, free, sinks, blocked


print("=== current maps in working tree ===")
for path in sorted((ROOT / "data" / "maps").glob("*.map")):
    w, h, free, sinks, blocked = geometry(path.read_text(encoding="utf-8"))
    print(f"{path.name:24} {w}x{h}  free={free}  sink(S)={sinks}  blocked={blocked}")
    added = sh(f'git log --diff-filter=A --format="%h %ad %an %s" --date=short -- "{path.relative_to(ROOT)}"')
    hist = sh(f'git log --format="%h %ad %an %s" --date=short -- "{path.relative_to(ROOT)}"')
    print(f"    added : {added.splitlines()[-1] if added else '(not in git history)'}")
    for line in hist.splitlines():
        print(f"    touch : {line}")

print("\n=== paper Table 2 ===")
for name, (w, h, tiles, v, e) in PAPER.items():
    print(f"{name:12} {w}x{h}  tiles={tiles}  #V={v}  #E={e}")

print("\n=== every map path ever seen in git history (all branches) ===")
print(sh('git log --all --diff-filter=A --name-only --format="COMMIT %h %ad %an %s" --date=short '
         '-- "*.map" "*maps*" | grep -v "^$" | head -80'))

print("\n=== branches / remotes ===")
print(sh("git branch -a --format='%(refname:short) %(objectname:short)' | head -30"))
print(sh("git remote -v"))
