#!/usr/bin/env python3
"""Scan every map blob ever committed and report distinct geometries.

Answers: does any historical map file have the traversable-cell counts printed
in Table 2 of the submitted manuscript (2649 / 10499 / 10200)?
"""
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WANT = {2649, 10499, 10200}


def sh(args):
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True).stdout


def geometry(text):
    body, height, width = [], None, None
    for line in text.splitlines():
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
    chars = defaultdict(int)
    for row in body:
        for ch in row:
            chars[ch] += 1
    return width, height, dict(chars)


# every (commit, path, blob) triple for map files across all refs
listing = sh(["git", "rev-list", "--all"]).split()
seen_blobs = {}
for commit in listing:
    tree = sh(["git", "ls-tree", "-r", commit, "--format=%(objectname) %(path)"])
    for row in tree.splitlines():
        if not row.strip():
            continue
        blob, path = row.split(" ", 1)
        if not path.endswith(".map"):
            continue
        seen_blobs.setdefault(blob, set()).add(path)

print(f"distinct map blobs in history: {len(seen_blobs)}\n")
rows = []
for blob, paths in seen_blobs.items():
    text = sh(["git", "cat-file", "-p", blob])
    w, h, chars = geometry(text)
    free = sum(v for k, v in chars.items() if k in ".GS")
    dots = chars.get(".", 0)
    rows.append((sorted(paths)[0], blob[:8], w, h, free, dots, chars))

hits = [r for r in rows if r[4] in WANT or r[5] in WANT]
print("=== blobs whose traversable count matches a Table 2 value ===")
for path, blob, w, h, free, dots, chars in sorted(hits):
    print(f"{path:56} {blob} {w}x{h} free={free} dots={dots} {dict(sorted(chars.items()))}")
if not hits:
    print("(none)")

print("\n=== all distinct large maps (free > 1000) ===")
for path, blob, w, h, free, dots, chars in sorted(rows, key=lambda r: (-r[4], r[0])):
    if free > 1000:
        print(f"{path:56} {blob} {w}x{h:<5} free={free:6} dots={dots:6} "
              f"{ {k: v for k, v in sorted(chars.items()) if k not in '.'} }")
