# tst_disperse.py
# ------------------------------------------------------------
# 여러 랜덤 케이스로 Intersection → disperse_paths 검증
# tick_viewer로 매 케이스 결과를 인터랙티브 시각화
# ------------------------------------------------------------
import argparse
import random
from typing import Dict, List, Tuple

from utils.Intersection import Intersection
from tick_viewer import play_ticks_curses


# --- 최소 AMR 스텁 (Intersection.register_amr 가 요구하는 필드만) --- #
class AMR:
    def __init__(self, aid: int, pos: Tuple[int, int]):
        self.id = aid
        self.pos = pos
        self.path: List[Tuple[int, int]] = []

    def set_path(self, path: List[Tuple[int, int]]):
        self.path = path


# --- 헬퍼들 (tst.py에서 가져온 것 + 약간 확장) --- #
def build_path_through_center(I: Intersection, start_pos: Tuple[int, int], exit_dir: str) -> List[Tuple[int, int]]:
    """register_amr가 exit_arm을 추출할 수 있도록 center를 경유시키는 경로"""
    center = (I.center_x, I.center_y)
    exit_front = I.lane_coords[exit_dir][0]
    return [start_pos, center, exit_front]


def register(I: Intersection, amr: AMR, exit_dir: str):
    amr.path = build_path_through_center(I, amr.pos, exit_dir)
    I.register_amr(amr)


def seed_paths_from_intent(I: Intersection):
    """현재 위치로 self.paths 시드 (disperse_paths가 self.paths를 덮어쓸 수 있게 초기화)"""
    I.paths = {}
    for aid, rec in I.amr_intent_map.items():
        a = rec.get("amr_obj")
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
        amr = rec.get("amr_obj")
        if not amr:
            continue
        exit_dir = rec.get("exit_arm")
        pos2info[amr.pos] = (aid, exit_dir)

    for d, coords in I.lane_coords.items():
        lanes[d] = [pos2info.get(p) for p in coords]
    return lanes


def pprint_lanes(title: str, lanes: Dict[str, List[tuple | None]]):
    def cell(v):
        if v is None:
            return "."
        aid, exit_dir = v
        return f"{aid}{exit_dir}"

    def row(vals):
        return "[" + ", ".join(cell(v) for v in vals) + "]"

    parts = []
    for d in "NESW":
        if d in lanes:
            parts.append(f"{d}:{row(lanes[d])}")
    print(f"{title}:  " + " | ".join(parts))


def all_coords_ok(I: Intersection, paths: Dict[int, List[Tuple[int, int]]]):
    okset = set(I.all_lane_coords)
    okset.add((I.center_x, I.center_y))
    for aid, p in paths.items():
        for xy in p:
            if xy not in okset:
                return False, (aid, xy)
    return True, None


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def steps_ok(paths: Dict[int, List[Tuple[int, int]]]):
    for aid, p in paths.items():
        for i in range(len(p) - 1):
            if manhattan(p[i], p[i + 1]) > 1:
                return False, (aid, p[i], p[i + 1])
    return True, None


def _fmt_xy(xy, center=None):
    if xy is None:
        return "."
    if center and xy == center:
        return "C"
    x, y = xy
    return f"({x},{y})"


def print_paths_tickwise(paths: Dict[int, List[Tuple[int, int]]], *, center=None, pad="repeat"):
    """행=틱, 열=AMR ID로 정렬 출력"""
    if not paths:
        print("\n[tickwise] (empty)")
        return
    amr_ids = sorted(paths.keys())
    max_len = max(len(p) for p in paths.values())

    table = {aid: [] for aid in amr_ids}
    for aid in amr_ids:
        p = paths[aid]
        if p and pad == "repeat":
            padded = p + [p[-1]] * (max_len - len(p))
        else:
            padded = p + [None] * (max_len - len(p))
        table[aid] = [_fmt_xy(xy, center=center) for xy in padded]

    col_w = max(
        max(len(s) for aid in amr_ids for s in table[aid]),
        max(len(str(aid)) for aid in amr_ids),
    )
    print("\n[tickwise]")
    print("tick ".ljust(6) + " ".join(f"{aid:>{col_w}}" for aid in amr_ids))
    print("-" * (6 + (col_w + 1) * len(amr_ids)))
    for t in range(max_len):
        row = " ".join(f"{table[aid][t]:>{col_w}}" for aid in amr_ids)
        print(f"T{t:02d}  {row}")


# --- disperse_paths 검증용 헬퍼들 --- #
def infer_arm(I: Intersection, pos: Tuple[int, int] | None) -> str | None:
    """좌표가 어느 팔 / center에 있는지 arm label 반환 ('N','E','S','W','C' 또는 None)"""
    if pos is None:
        return None
    cx, cy = I.center_x, I.center_y
    if pos == (cx, cy):
        return "C"
    for d, coords in I.lane_coords.items():
        if pos in coords:
            return d
    return None


def final_lanes_from_paths(I: Intersection, paths: Dict[int, List[Tuple[int, int]]]) -> Dict[str, List[int | None]]:
    """마지막 tick 기준 각 레인에 어떤 AMR이 있는지 인덱스별로 복원"""
    finals = {aid: p[-1] for aid, p in paths.items()}
    lanes: Dict[str, List[int | None]] = {}
    for d, coords in I.lane_coords.items():
        lanes[d] = [None] * len(coords)
    for aid, pos in finals.items():
        for d, coords in I.lane_coords.items():
            if pos in coords:
                idx = coords.index(pos)
                assert lanes[d][idx] is None, f"collision at {d}[{idx}] between AMRs"
                lanes[d][idx] = aid
                break
    return lanes


# --- 랜덤 케이스 러너 (disperse_paths) --- #
def run_random_disperse_case(
    case_idx: int,
    n_amrs: int,
    lenN: int,
    lenE: int,
    lenS: int,
    lenW: int,
    seed: int,
    view: bool,
    max_move: int | None,
):
    print(f"\n== Case #{case_idx}: random disperse_paths (seed={seed}) ==")
    if seed != 0:
        random.seed(seed)

    # 교차로 생성 (중앙 (10,10); 팔 길이는 인자)
    I = Intersection(
        intersection_data=(10, 10, lenN, lenE, lenS, lenW),
        present_dirs={"N", "E", "S", "W"},
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

    # input_dir 하나 뽑고, 그 레인의 AMR 수만큼 이동 상한 설정
    input_dir = random.choice(list(I.dirs))
    initial_lanes = snapshot_lanes(I)
    n_in_input = sum(1 for cell in initial_lanes.get(input_dir, []) if cell is not None)
    if max_move is not None:
        num_amrs = min(max_move, n_in_input)
    else:
        num_amrs = random.randint(0, n_in_input) if n_in_input > 0 else 0

    print(f"disperse_paths input_dir={input_dir}, num_amrs={num_amrs}")

    # 실행
    target_lanes, paths = I.disperse_paths(input_dir=input_dir, num_amrs=num_amrs)

    # I.paths가 반환값과 일치한다고 가정
    assert paths is I.paths or paths == I.paths

    # 검증 1: 좌표 범위 / 한 틱 이동 거리
    ok, detail = all_coords_ok(I, paths)
    if not ok:
        raise AssertionError(f"⚠️ invalid coord found: {detail}")
    ok2, detail2 = steps_ok(paths)
    if not ok2:
        raise AssertionError(f"⚠️ 한 틱에 2칸 이상 이동: {detail2}")

    # 검증 2: 모든 path 길이 동일
    lens = {len(p) for p in paths.values()}
    assert len(lens) == 1, f"paths length mismatch: {lens}"

    # 검증 3: 최종 레인 배치가 target_lanes와 일치하는지
    final_lanes = final_lanes_from_paths(I, paths)
    for d in I.dirs:
        tgt = target_lanes.get(d, [])
        fin = final_lanes.get(d, [])
        assert len(tgt) == len(fin), f"lane length mismatch at {d}: {len(tgt)} vs {len(fin)}"
        for i, (aid_tgt, aid_fin) in enumerate(zip(tgt, fin)):
            if aid_tgt is None:
                # target이 None이면 final도 None이거나 AMR이 없어야 한다
                assert aid_fin is None, f"{d}[{i}] expected None, got AMR {aid_fin}"
            else:
                assert aid_fin == aid_tgt, f"{d}[{i}] expected AMR {aid_tgt}, got {aid_fin}"

    # 검증 4: arm 이동 제약
    #   - input_dir / center 출발이 아닌 AMR은 팔이 바뀌면 안 됨
    center_xy = (I.center_x, I.center_y)
    final_pos = {aid: p[-1] for aid, p in paths.items()}
    initial_pos = {aid: p[0] for aid, p in paths.items()}

    initial_arm = {aid: infer_arm(I, pos) for aid, pos in initial_pos.items()}
    final_arm = {aid: infer_arm(I, pos) for aid, pos in final_pos.items()}

    moved_from_input = 0
    for aid in paths.keys():
        ia = initial_arm[aid]
        fa = final_arm[aid]
        if ia == input_dir:
            if fa is not None and fa != input_dir:
                moved_from_input += 1
        elif ia == "C":
            # center 출발인 경우는 아무 팔로 가도 OK
            pass
        else:
            # 다른 팔에서 출발한 AMR은 팔이 바뀌면 안 됨
            assert ia == fa, f"AMR {aid} moved from arm {ia} to {fa}, but only {input_dir}/center should move"

    assert moved_from_input <= num_amrs, (
        f"moved {moved_from_input} AMRs from {input_dir}, "
        f"but requested num_amrs={num_amrs}"
    )

    print_paths_tickwise(paths, center=center_xy, pad="repeat")

    # 뷰어로 확인 (q로 종료 → 다음 케이스 진행)
    if view:
        play_ticks_curses(I)

    # 요약
    for aid, p in sorted(paths.items()):
        print(f"AMR {aid:>2} len={len(p)} tail={p}")

    print("✅ disperse_paths case passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=1, help="몇 개의 랜덤 케이스를 돌릴지")
    ap.add_argument("--amrs", type=int, default=15, help="케이스당 AMR 수")
    ap.add_argument("--lenN", type=int, default=5, help="N 레인 길이")
    ap.add_argument("--lenE", type=int, default=5, help="E 레인 길이")
    ap.add_argument("--lenS", type=int, default=5, help="S 레인 길이")
    ap.add_argument("--lenW", type=int, default=5, help="W 레인 길이")
    ap.add_argument("--seed", type=int, default=0, help="전역 시드(케이스별로 +i)")
    ap.add_argument("--moves", type=int, default=3, help="각 케이스에서 최대 몇 대까지 옮길지 상한 (없으면 랜덤)")
    ap.add_argument("--no-view", action="store_true", help="tick_viewer 생략")
    args = ap.parse_args()

    for i in range(args.cases):
        run_random_disperse_case(
            case_idx=i + 1,
            n_amrs=args.amrs,
            lenN=args.lenN,
            lenE=args.lenE,
            lenS=args.lenS,
            lenW=args.lenW,
            seed=args.seed + i if args.seed != 0 else 0,
            view=not args.no_view,
            max_move=args.moves,
        )

    print("\nAll disperse_paths random cases finished ✅")


if __name__ == "__main__":
    main()
