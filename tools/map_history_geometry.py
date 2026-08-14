#!/usr/bin/env python3
"""Measure geometry of historical map blobs so Table 2 can be traced to a commit."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TARGETS = [
    ("1bf5a35", "data/maps/warehouse_10_20.map"),
    ("1bf5a35", "data/maps/warehouse_20_40.map"),
    ("1bf5a35", "data/maps/cross_3030.map"),
    ("ec999d7", "assets/warehouse-10-20/warehouse-10-20.map"),
    ("ec999d7", "assets/warehouse-20-40/warehouse-20-40.map"),
    ("ec999d7", "assets/cross-30-30/cross-30-30.map"),
    ("e206cba", "problems/cross_3030/maps/cross_3030.map"),
    ("0a9f995", "problems/cross/maps/warehouse-10-20-10-2-1.map"),
    ("0a9f995", "problems/cross/maps/warehouse-20-40-10-2-1.map"),
    ("github/main", "data/maps/cross_3030.map"),
    ("github/main", "data/maps/warehouse_10_20.map"),
    ("github/main", "data/maps/warehouse_20_40.map"),
]


def blob(rev, path):
    out = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=ROOT,
                         capture_output=True, text=True)
    return out.stdout if out.returncode == 0 else None


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
    chars = {}
    for row in body:
        for ch in row:
            chars[ch] = chars.get(ch, 0) + 1
    free = sum(v for k, v in chars.items() if k in ".GS")
    return width, height, free, chars


print(f"{'rev':12} {'path':52} {'dims':>10} {'free':>7}  charset")
print("-" * 110)
for rev, path in TARGETS:
    text = blob(rev, path)
    if text is None:
        print(f"{rev:12} {path:52} {'(absent)':>10}")
        continue
    w, h, free, chars = geometry(text)
    cs = " ".join(f"{k!r}:{v}" for k, v in sorted(chars.items(), key=lambda x: -x[1]))
    print(f"{rev:12} {path:52} {w}x{h:<6} {free:7}  {cs}")

print("\npaper Table 2: Standard 1 161x63 tiles=2649 | Standard 2 321x123 tiles=10499 | Square 1 267x187 tiles=10200")
