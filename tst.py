# test_prestage_paths_real_print_paths.py
# ------------------------------------------------------------
# 실제 Intersection을 사용한 프리-스테이지 검증 + AMR별 경로 출력
# ------------------------------------------------------------

from typing import Dict, List, Tuple

# ⬇️ 프로젝트 경로에 맞게 수정
from utils.Intersection import Intersection

# --- 최소 AMR 스텁(Intersection은 amr_obj.pos 및 set_path만 사용) ---
class AMR:
    def __init__(self, aid, pos):
        self.id = aid
        self.pos = pos
        self.path = []
    def set_path(self, path):
        self.path = path

# ── pretty helpers ──────────────────────────────────────────
def row(vals: List[int | None]) -> str:
    """[., 2, 3] 형태로 출력"""
    return "[" + ", ".join("." if v is None else str(v) for v in vals) + "]"

def pprint_lanes(title, lanes: Dict[str, List[int | None]]):
    parts = []
    for d in "NESW":
        if d in lanes:
            parts.append(f"{d}:{row(lanes[d])}")
    print(f"{title}:  " + " | ".join(parts))

def print_paths(title: str, paths: Dict[int, List[Tuple[int, int]]]):
    """AMR별 경로를 보기 좋게 출력"""
    print(f"\n{title}")
    if not paths:
        print("  (empty)")
        return
    for aid in sorted(paths.keys()):
        p = paths[aid]
        seq = " -> ".join(f"({x},{y})" for (x, y) in p)
        print(f"  AMR {aid:>3}  len={len(p)}  {seq}")

# ── 시나리오 ────────────────────────────────────────────────
def run_scenario_1():
    print("\n== Scenario 1: center→exit(front) & compress ==")
    # 교차로 생성(near→far: 3칸씩)
    I = Intersection(
        intersection_data=(10,10,3,3,3,3),
        neighbors_map={},
        present_dirs={'N','E','S','W'},
    )
    cx, cy = I.center_x, I.center_y

    # 배치: N:[.,2,3], E:[4,.,5], S:[.,.,6], W:[.,.,.], center:1 (exit=E)
    amrs = {
        1: AMR(1, (cx,cy)),                         # center
        2: AMR(2, I.lane_coords['N'][1]),
        3: AMR(3, I.lane_coords['N'][2]),
        4: AMR(4, I.lane_coords['E'][0]),
        5: AMR(5, I.lane_coords['E'][2]),
        6: AMR(6, I.lane_coords['S'][2]),
    }
    I.amr_intent_map.clear()
    for aid, a in amrs.items():
        I.amr_intent_map[aid] = {'amr_obj': a, 'exit_arm': 'E'}

    ret = I.build_prestage_paths()
    # (target_lanes, paths) 형태를 가정
    target, paths = ret

    pprint_lanes("target  ", target)
    print_paths("paths    ", paths)

def run_scenario_2():
    print("\n== Scenario 2: exit full → min-occupancy(front) ==")
    I = Intersection(
        intersection_data=(20,20,3,3,3,3),
        neighbors_map={},
        present_dirs={'N','E','S','W'},
    )
    cx, cy = I.center_x, I.center_y

    # E 만실, center exit=E → 최소 점유 팔(S) front에 배치
    amrs = {
        1: AMR(1, (cx,cy)),                         # center
        7: AMR(7, I.lane_coords['E'][0]),
        8: AMR(8, I.lane_coords['E'][1]),
        9: AMR(9, I.lane_coords['E'][2]),
        10: AMR(10, I.lane_coords['N'][0]),
        11: AMR(11, I.lane_coords['N'][1]),
        12: AMR(12, I.lane_coords['S'][0]),
        13: AMR(13, I.lane_coords['W'][0]),
    }
    I.amr_intent_map.clear()
    for aid, a in amrs.items():
        I.amr_intent_map[aid] = {'amr_obj': a, 'exit_arm': 'E'}

    ret = I.build_prestage_paths()
    target, paths = ret

    pprint_lanes("target  ", target)
    print_paths("paths    ", paths)

if __name__ == "__main__":
    run_scenario_1()
    run_scenario_2()
    print("\nAll tests passed ✅")
