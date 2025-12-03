# tst.py
# ------------------------------------------------------------
# 여러 랜덤 케이스로 Intersection → plan_action → actions_to_paths 검증
# tick_viewer로 매 케이스 결과를 인터랙티브 시각화
# ------------------------------------------------------------
import argparse
import random
from typing import Dict, List, Tuple

from utils.Intersection import Intersection
from tick_viewer import play_ticks_curses


# --- 최소 AMR 스텁 (Intersection.register_amr 가 요구하는 필드만) ---
class AMR:
    def __init__(self, aid: int, pos: Tuple[int,int]):
        self.id = aid
        self.pos = pos
        self.path: List[Tuple[int,int]] = []

    def set_path(self, path: List[Tuple[int,int]]):
        self.path = path


# --- 헬퍼들 ---
def build_path_through_center(I: Intersection, start_pos: Tuple[int,int], exit_dir: str) -> List[Tuple[int,int]]:
    """register_amr가 exit_arm을 추출할 수 있도록 center를 경유시키는 경로"""
    center = (I.center_x, I.center_y)
    exit_front = I.lane_coords[exit_dir][0]
    return [start_pos, center, exit_front]

def register(I: Intersection, amr: AMR, exit_dir: str):
    amr.path = build_path_through_center(I, amr.pos, exit_dir)
    I.register_amr(amr)

def seed_paths_from_intent(I: Intersection):
    """현재 위치로 self.paths 시드 (actions_to_paths가 self.paths를 사용)"""
    I.paths = {}
    for aid, rec in I.amr_intent_map.items():
        a = rec.get('amr_obj')
        if not a: 
            continue
        pos = getattr(a, "pos", None)
        if pos is None:
            continue
        I.paths[aid] = [pos]

def snapshot_lanes(I: Intersection) -> Dict[str, List[tuple | None]]:
    """각 레인 셀에 (amr_id, exit_dir) 또는 None 저장"""
    lanes = {}
    pos2info = {}
    for aid, rec in I.amr_intent_map.items():
        amr = rec.get('amr_obj')
        if not amr:
            continue
        exit_dir = rec.get('exit_arm')
        pos2info[amr.pos] = (aid, exit_dir)

    for d, coords in I.lane_coords.items():
        lanes[d] = [pos2info.get(p) for p in coords]
    return lanes

def pprint_lanes(title: str, lanes: Dict[str, List[tuple | None]]):
    def cell(v):
        if v is None:
            return "."
        aid, exit_dir = v
        # 모양은 취향대로: "3N" 또는 "3(N)" 등
        return f"{aid}{exit_dir}"

    def row(vals):
        return "[" + ", ".join(cell(v) for v in vals) + "]"

    parts = []
    for d in "NESW":
        if d in lanes:
            parts.append(f"{d}:{row(lanes[d])}")
    print(f"{title}:  " + " | ".join(parts))


def all_coords_ok(I: Intersection, paths: Dict[int, List[Tuple[int,int]]]):
    okset = set(I.all_lane_coords)
    okset.add((I.center_x, I.center_y))
    for aid, p in paths.items():
        for xy in p:
            if xy not in okset:
                return False, (aid, xy)
    return True, None

def manhattan(a,b): 
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def steps_ok(paths: Dict[int, List[Tuple[int,int]]]):
    for aid, p in paths.items():
        for i in range(len(p)-1):
            if manhattan(p[i], p[i+1]) > 1:
                return False, (aid, p[i], p[i+1])
    return True, None

def _fmt_xy(xy, center=None):
    if xy is None:
        return "."
    if center and xy == center:
        return "C"
    x, y = xy
    return f"({x},{y})"

def print_paths_tickwise(paths: Dict[int, List[Tuple[int,int]]], *, center=None, pad="repeat"):
    """행=틱, 열=AMR ID로 정렬 출력"""
    if not paths:
        print("\n[tickwise] (empty)")
        return
    amr_ids = sorted(paths.keys())
    max_len = max(len(p) for p in paths.values())

    table = {aid: [] for aid in amr_ids}
    for aid in amr_ids:
        p = paths[aid]
        padded = p + [p[-1]] * (max_len - len(p)) if (p and pad == "repeat") else (p + [None] * (max_len - len(p)))
        table[aid] = [_fmt_xy(xy, center=center) for xy in padded]

    col_w = max(max(len(s) for aid in amr_ids for s in table[aid]), max(len(str(aid)) for aid in amr_ids))
    print("\n[tickwise]")
    print("tick ".ljust(6) + " ".join(f"{aid:>{col_w}}" for aid in amr_ids))
    print("-" * (6 + (col_w + 1) * len(amr_ids)))
    for t in range(max_len):
        row = " ".join(f"{table[aid][t]:>{col_w}}" for aid in amr_ids)
        print(f"T{t:02d}  {row}")


# --- 랜덤 케이스 러너 ---
def run_random_case(case_idx: int, n_amrs: int, lenN: int, lenE: int, lenS: int, lenW: int, seed: int, view: bool):
    print(f"\n== Case #{case_idx}: random actions_to_paths (seed={seed}) ==")
    if seed != 0:
        random.seed(seed)

    # 교차로 생성 (중앙 (10,10); 팔 길이는 인자)
    I = Intersection(
        intersection_data=(10, 10, lenN, lenE, lenS, lenW),
        present_dirs={'N','E','S','W'},
    )

    # 시작 위치 후보(레인 전체) 준비
    lane_cells = []
    for d in I.dirs:
        lane_cells.extend(I.lane_coords[d])
    random.shuffle(lane_cells)

    # AMR 수 제한(레인 전체 칸 수 초과 방지)
    n_cap = min(n_amrs, len(lane_cells))
    chosen_starts = lane_cells[:n_cap]

    # 무작위 AMR 생성 & 등록(출구 방향도 랜덤)
    for idx, start in enumerate(chosen_starts, start=1):
        exit_dir = random.choice(list(I.dirs))
        amr = AMR(idx, start)
        register(I, amr, exit_dir)

    # paths 시드
    seed_paths_from_intent(I)

    print("initial lanes:")
    pprint_lanes("lanes   ", snapshot_lanes(I))

    # 실행
    I.actions_to_paths()

    # 검증
    ok, detail = all_coords_ok(I, I.paths)
    if not ok:
        print(f"⚠️ invalid coord found: {detail}")
    ok2, detail2 = steps_ok(I.paths)
    assert ok2, f"한 틱에 2칸 이상 이동: {detail2}"

    print_paths_tickwise(I.paths, center=(I.center_x, I.center_y), pad="repeat")

    # 뷰어로 확인 (q로 종료 → 다음 케이스 진행)
    if view:
        play_ticks_curses(I)

    # 요약
    for aid, p in sorted(I.paths.items()):
        print(f"AMR {aid:>2} len={len(p)} tail={p}")

    print("✅ case passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=1, help="몇 개의 랜덤 케이스를 돌릴지")
    ap.add_argument("--amrs", type=int, default=15, help="케이스당 AMR 수")
    ap.add_argument("--lenN", type=int, default=5)
    ap.add_argument("--lenE", type=int, default=5)
    ap.add_argument("--lenS", type=int, default=5)
    ap.add_argument("--lenW", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0, help="전역 시드(케이스별로 +i)")
    ap.add_argument("--no-view", action="store_true", help="tick_viewer 생략")
    args = ap.parse_args()

    for i in range(args.cases):
        run_random_case(
            case_idx=i+1,
            n_amrs=args.amrs,
            lenN=args.lenN, lenE=args.lenE, lenS=args.lenS, lenW=args.lenW,
            seed=args.seed,
            view=not args.no_view
        )

    print("\nAll random cases finished ✅")


if __name__ == "__main__":
    main()
