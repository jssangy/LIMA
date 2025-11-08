# test_actions_to_paths_integration.py
# ------------------------------------------------------------
# Intersection 전체 흐름(등록 → plan_action → actions_to_paths) 통합 테스트
# ------------------------------------------------------------
import random
from typing import Dict, List, Tuple
from utils.Intersection import Intersection
from tick_viewer import play_ticks_curses

# --- 최소 AMR 스텁 ---
class AMR:
    def __init__(self, aid, pos):
        self.id = aid
        self.pos = pos
        self.path = []
    def set_path(self, path):
        self.path = path

# --- 헬퍼들 ---
def build_path_through_center(I: Intersection, start_pos, exit_dir: str):
    """register_amr가 exit_arm을 추출할 수 있도록 center를 경유하는 간단 경로 생성"""
    center = (I.center_x, I.center_y)
    exit_front = I.lane_coords[exit_dir][0]
    return [start_pos, center, exit_front]

def register(I: Intersection, amr: AMR, exit_dir: str):
    amr.path = build_path_through_center(I, amr.pos, exit_dir)
    I.register_amr(amr)

def seed_paths_from_intent(I: Intersection):
    """현재 위치로 self.paths 시드 (actions_to_paths가 self.paths를 사용하므로 필수)"""
    I.paths = {}
    for aid, rec in I.amr_intent_map.items():
        a = rec.get('amr_obj')
        if a is None: 
            continue
        pos = getattr(a, "pos", None)
        if pos is None:
            continue
        I.paths[aid] = [pos]

def snapshot_lanes(I: Intersection):
    lanes = {}
    pos2id = {rec['amr_obj'].pos: aid for aid, rec in I.amr_intent_map.items() if rec.get('amr_obj')}
    for d, coords in I.lane_coords.items():
        lanes[d] = [pos2id.get(p) for p in coords]
    return lanes

def pprint_lanes(title, lanes: Dict[str, List[int|None]]):
    def row(vals):
        return "[" + ", ".join("." if v is None else str(v) for v in vals) + "]"
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

def manhattan(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])

def steps_ok(paths):
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

def print_paths_tickwise(paths, *, center=None, pad="repeat"):
    """
    행=틱, 열=AMR ID 로 정렬 출력.
    pad="repeat"  -> 짧은 경로는 마지막 좌표를 반복해 패딩
    pad="dot"     -> 짧은 경로는 '.'로 패딩
    """
    if not paths:
        print("\n[tickwise] (empty)")
        return

    amr_ids = sorted(paths.keys())
    max_len = max(len(p) for p in paths.values())

    # 패딩 후 문자열로 변환
    table = {aid: [] for aid in amr_ids}
    for aid in amr_ids:
        p = paths[aid]
        if not p:
            padded = [None] * max_len
        else:
            if pad == "repeat":
                padded = p + [p[-1]] * (max_len - len(p))
            else:  # pad == "dot"
                padded = p + [None] * (max_len - len(p))
        table[aid] = [_fmt_xy(xy, center=center) for xy in padded]

    # 컬럼 너비 계산
    col_w = max(
        max(len(s) for aid in amr_ids for s in table[aid]),
        max(len(str(aid)) for aid in amr_ids)
    )

    # 헤더
    print("\n[tickwise]")
    print("tick ".ljust(6) + " ".join(f"{aid:>{col_w}}" for aid in amr_ids))
    print("-" * (6 + (col_w + 1) * len(amr_ids)))

    # 본문
    for t in range(max_len):
        row = " ".join(f"{table[aid][t]:>{col_w}}" for aid in amr_ids)
        print(f"T{t:02d}  {row}")



# --- 시나리오 3: 어려운 동작 ---
def run_scenario_hard():
    print("\n== Scenario 3: hard actions_to_paths ==")
    random.seed(2)

    I = Intersection(
        intersection_data=(10,10,3,3,3,3),
        neighbors_map={},
        present_dirs={'N','E','S','W'},
    )

    # 좌표 단축
    N0,N1,N2 = I.lane_coords['N']
    E0,E1,E2 = I.lane_coords['E']
    S0,S1,S2 = I.lane_coords['S']
    W0,W1,W2 = I.lane_coords['W']

    # AMR 생성 & 등록 (register_amr 실제 사용)
    a1 = AMR(1, N0)
    a2 = AMR(2, N1) 
    a3 = AMR(3, E0) 
    a4 = AMR(4, E1)  
    a5 = AMR(5, S0)  
    a6 = AMR(6, S1) 
    a7 = AMR(7, W0)  
    a8 = AMR(8, W1) 
    a9 = AMR(9, W2)
    register(I, a1, 'S')
    register(I, a2, 'E')
    register(I, a3, 'W')
    register(I, a4, 'S')
    register(I, a5, 'N')
    register(I, a6, 'W')
    register(I, a7, 'E')
    register(I, a8, 'N')
    register(I, a9, 'S')

    # 초기 self.paths 시드
    seed_paths_from_intent(I)

    print("initial lanes:")
    pprint_lanes("lanes   ", snapshot_lanes(I))

    # 실행: plan_action은 actions_to_paths 내부에서 호출됨
    I.actions_to_paths()

    # 검증
    paths = I.paths
    lens = {aid: len(p) for aid,p in paths.items()}

    ok, detail = all_coords_ok(I, paths)
    # assert ok, f"유효하지 않은 좌표 발견: {detail}"

    ok, detail = steps_ok(paths)
    assert ok, f"한 틱에 2칸 이상 이동: {detail}"

    center = (I.center_x, I.center_y)
    print_paths_tickwise(I.paths, center=center, pad="repeat")

    play_ticks_curses(I)

    for aid, p in I.paths.items():
        print(f"AMR {aid} path: {p}")

    print("✅ basic scenario passed")


if __name__ == "__main__":
    run_scenario_hard()
    print("\nAll integration tests passed ✅")
