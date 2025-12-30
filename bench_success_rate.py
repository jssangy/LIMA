import os
import re
import csv
import argparse
from statistics import mean
from collections import defaultdict
from typing import Optional, Tuple

# ---------- 파서 ----------
def read_success_rate_from_csv(path: str) -> float:
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        row = next(r, None)
        if row is None:
            raise ValueError("empty csv")
        if "success_rate" not in row:
            raise ValueError("missing column 'success_rate'")
        return float(row["success_rate"])

_SOLVED_RE = re.compile(r"(?m)^\s*#?\s*solved\s*[:=]\s*([0-9]*\.?[0-9]+)\s*$")

def read_success_rate_from_txt(path: str) -> float:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    m = _SOLVED_RE.search(txt)
    if not m:
        raise ValueError("cannot find 'solved[:=] <number>'")
    return float(m.group(1))

# ---------- 파일명에서 density / scen 추출 ----------
PAT_A = re.compile(r"^(?P<map>.+)_(?P<den>\d+)_s(?P<scen>\d+)\.(?P<ext>csv|txt)$")
PAT_B = re.compile(r"^res_edge_(?P<edge>\d+)_Den(?P<den>\d+)_N\d+\.txt$")  # lacam 옛 이름도 지원

def extract_den_scen(filename: str) -> Optional[Tuple[int, int]]:
    m = PAT_A.match(filename)
    if m:
        return int(m.group("den")), int(m.group("scen"))
    m = PAT_B.match(filename)
    if m:
        edge = int(m.group("edge"))
        den = int(m.group("den"))
        scen = edge - 1
        return den, scen
    return None

# ---------- 맵 자동 탐지 ----------
def detect_maps(assets_root: str):
    maps = []
    if not os.path.isdir(assets_root):
        return maps
    for name in os.listdir(assets_root):
        p = os.path.join(assets_root, name)
        if not os.path.isdir(p):
            continue
        if os.path.isfile(os.path.join(p, f"{name}.map")):
            maps.append(name)
    return sorted(maps)

def summarize(values):
    return {"n": len(values), "mean": mean(values), "min": min(values), "max": max(values)}

def write_csv(path: str, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--assets-root", default="assets")
    p.add_argument("--maps", nargs="*", default=None, help="비우면 assets 아래 map 자동 탐지")
    p.add_argument("--densities", nargs="*", type=int, default=[1, 5, 10, 20, 30, 40, 50, 60])
    p.add_argument("--scens", nargs="*", type=int, default=list(range(10)))
    p.add_argument("--out", default=None, help="기본: assets/summary_sr_by_alg_density_map.csv")
    args = p.parse_args()

    maps = args.maps if args.maps else detect_maps(args.assets_root)
    if not maps:
        raise SystemExit("No maps found. (Check --assets-root or --maps)")

    alg_cfg = {
        "cbs":   {"subdir": "results_cbs",  "kind": "csv"},
        "lima":  {"subdir": "results_lima", "kind": "csv"},
        "lacam": {"subdir": "results_lacam","kind": "txt"},
        "pibt":  {"subdir": "results_pibt", "kind": "txt"},
    }

    # data[map][alg][den] = list of success_rate
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    missing = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for map_name in maps:
        for alg, cfg in alg_cfg.items():
            rdir = os.path.join(args.assets_root, map_name, cfg["subdir"])
            if not os.path.isdir(rdir):
                continue

            for fn in os.listdir(rdir):
                ex = extract_den_scen(fn)
                if ex is None:
                    continue
                den, scen = ex
                if den not in args.densities or scen not in args.scens:
                    continue
                fp = os.path.join(rdir, fn)

                try:
                    sr = read_success_rate_from_csv(fp) if cfg["kind"] == "csv" else read_success_rate_from_txt(fp)
                    data[map_name][alg][den].append(sr)
                except Exception as e:
                    print(f"[WARN] {map_name} {alg} failed to read {fp}: {e}")

            # 기대 파일(den×scen) 대비 missing 계산
            expected = len(args.scens)
            for den in args.densities:
                got = len(data[map_name][alg].get(den, []))
                missing[map_name][alg][den] = max(0, expected - got)

    # (map, algorithm, density) 요약
    rows = []
    for map_name in sorted(data.keys()):
        for alg in sorted(data[map_name].keys()):
            for den in args.densities:
                vals = data[map_name][alg].get(den, [])
                if not vals:
                    rows.append({
                        "map": map_name,
                        "algorithm": alg,
                        "density": den,
                        "n": 0,
                        "missing": missing[map_name][alg][den],
                        "mean_success_rate": "",
                        "min_success_rate": "",
                        "max_success_rate": "",
                    })
                    continue
                s = summarize(vals)
                rows.append({
                    "map": map_name,
                    "algorithm": alg,
                    "density": den,
                    "n": s["n"],
                    "missing": missing[map_name][alg][den],
                    "mean_success_rate": f"{s['mean']:.6f}",
                    "min_success_rate": f"{s['min']:.6f}",
                    "max_success_rate": f"{s['max']:.6f}",
                })

    out_path = args.out or os.path.join(args.assets_root, "summary_sr.csv")
    write_csv(
        out_path,
        rows,
        fieldnames=[
            "map", "algorithm", "density", "n", "missing",
            "mean_success_rate", "min_success_rate", "max_success_rate"
        ],
    )
    print(f"[WRITE] {out_path}")

if __name__ == "__main__":
    main()
