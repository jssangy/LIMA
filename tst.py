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

def row(vals):
    return "[" + ", ".join("." if v is None else str(v) for v in vals) + "]"

def pprint_lanes(title, lanes: Dict[str, List[int]]):
    parts = []
    for d in "NESW":
        if d in lanes:
            parts.append(f"{d}:{row(lanes[d])}")
    print(f"{title}:  " + " | ".join(parts))

def manhattan(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])

def print_paths(paths: Dict[int, List[Tuple[int,int]]], *, title="AMR Paths"):
    print(f"\n{title}:")
    for aid in sorted(paths.keys()):
        p = paths[aid]
        arrow = " -> "
        s = arrow.join(f"({x},{y})" for (x,y) in p)
        print(f"  AMR {aid:>3}  len={len(p)}  {s}")

def collect_amr_paths_from_intent(I: Intersection) -> Dict[int, List[Tuple[int,int]]]:
    out = {}
    for aid, rec in I.amr_intent_map.items():
        a = rec.get('amr_obj')
        if a is None: 
            continue
        path = getattr(a, "path", None)
        if isinstance(path, list) and len(path) >= 1:
            out[aid] = path
    return out

def verify_paths_sync(I: Intersection, target_lanes, paths: Dict[int, List[Tuple[int,int]]], max_steps: int):
    # 1) 모든 경로 길이 동일
    expected_len = max_steps + 1
    lens = {aid: len(p) for aid, p in paths.items()}
    assert len(set(lens.values())) == 1, f"경로 길이 불일치: {lens}"
    one_len = next(iter(lens.values()))
    assert one_len == expected_len, f"경로 길이({one_len}) != max_steps+1({expected_len})"

    # 2) 한 틱 이동은 1칸(또는 정지)
    for aid, p in paths.items():
        for i in range(len(p)-1):
            assert manhattan(p[i], p[i+1]) <= 1, f"AMR {aid}가 한 틱에 2칸 이상 이동: {p[i]} -> {p[i+1]}"

    # 3) 최종 좌표가 target_lanes의 목표 좌표와 일치
    target_pos = {}
    for d in "NESW":
        if d not in target_lanes: continue
        coords = I.lane_coords[d]  # near→far
        for i, aid in enumerate(target_lanes[d]):
            if aid is not None:
                target_pos[aid] = coords[i]
    for aid, p in paths.items():
        if aid in target_pos:
            assert p[-1] == target_pos[aid], f"AMR {aid} 최종 좌표 불일치: {p[-1]} != {target_pos[aid]}"

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

    # 반환 형태 호환(2-리턴 / 4-리턴)
    if isinstance(ret, tuple) and len(ret) == 4:
        lanes, target, paths, max_steps = ret
        pprint_lanes("lanes   ", lanes)
        pprint_lanes("target  ", target)

        # 기대 target
        expect = {
            'N': [2,3,None],
            'E': [1,4,5],
            'S': [6,None,None],
            'W': [None,None,None]
        }
        for d in expect:
            assert target[d] == expect[d], f"{d} mismatch: {target[d]} != {expect[d]}"

        verify_paths_sync(I, target, paths, max_steps)
        print_paths(paths, title="AMR Paths (Scenario 1)")
        print("✅ Scenario 1 passed (max_steps=", max_steps, ")")
        print(I.paths)
    elif isinstance(ret, tuple) and len(ret) == 2:
        lanes, target = ret
        pprint_lanes("lanes   ", lanes)
        pprint_lanes("target  ", target)
        # 경로는 Intersection이 내부에서 set_path 했을 수도 있다 → 수집해서 출력
        amr_paths = collect_amr_paths_from_intent(I)
        if amr_paths:
            print_paths(amr_paths, title="AMR Paths from amr_obj (Scenario 1)")
        else:
            print("⚠️ paths/max_steps가 반환되지 않았고 amr_obj.path도 비어 있습니다.")
        # 기대 target 검증
        expect = {
            'N': [2,3,None],
            'E': [1,4,5],
            'S': [6,None,None],
            'W': [None,None,None]
        }
        for d in expect:
            assert target[d] == expect[d], f"{d} mismatch: {target[d]} != {expect[d]}"
        print("⚠️ 동기화·최종좌표 검증은 paths가 없어 생략되었습니다.")
    else:
        raise AssertionError("build_prestage_paths 반환 형식이 예상과 다릅니다.")

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

    if isinstance(ret, tuple) and len(ret) == 4:
        lanes, target, paths, max_steps = ret
        pprint_lanes("lanes   ", lanes)
        pprint_lanes("target  ", target)

        expect = {
            'N': [10,11,None],
            'E': [7,8,9],
            'S': [1,12,None],
            'W': [13,None,None]
        }
        for d in expect:
            assert target[d] == expect[d], f"{d} mismatch: {target[d]} != {expect[d]}"

        verify_paths_sync(I, target, paths, max_steps)
        print_paths(paths, title="AMR Paths (Scenario 2)")
        print("✅ Scenario 2 passed (max_steps=", max_steps, ")")
        print(I.paths)
    elif isinstance(ret, tuple) and len(ret) == 2:
        lanes, target = ret
        pprint_lanes("lanes   ", lanes)
        pprint_lanes("target  ", target)
        amr_paths = collect_amr_paths_from_intent(I)
        if amr_paths:
            print_paths(amr_paths, title="AMR Paths from amr_obj (Scenario 2)")
        else:
            print("⚠️ paths/max_steps가 반환되지 않았고 amr_obj.path도 비어 있습니다.")
        expect = {
            'N': [10,11,None],
            'E': [7,8,9],
            'S': [1,12,None],
            'W': [13,None,None]
        }
        for d in expect:
            assert target[d] == expect[d], f"{d} mismatch: {target[d]} != {expect[d]}"
        print("⚠️ 동기화·최종좌표 검증은 paths가 없어 생략되었습니다.")
    else:
        raise AssertionError("build_prestage_paths 반환 형식이 예상과 다릅니다.")

if __name__ == "__main__":
    run_scenario_1()
    run_scenario_2()
    print("\nAll tests passed ✅")
