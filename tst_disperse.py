# tst_disperse.py
# ------------------------------------------------------------
# Intersection.disperse_paths 검증용 랜덤 테스트
#  - 경로 형식 (좌표 범위, 1틱 1칸 이동, 길이 동기화)
#  - target_lanes와 최종 위치 일치 여부
#  - arm 이동 제약 (input_dir / center 외 팔은 arm 고정)
#  - "탈출 가능한 AMR" + input_dir 팔에 그대로 남은 AMR 제거 로직 검증
# ------------------------------------------------------------
import argparse
import random
from typing import Dict, List, Tuple, Optional

from utils.Intersection import Intersection
try:
    from tick_viewer import play_ticks_curses
    HAS_VIEWER = True
except ImportError:
    HAS_VIEWER = False


# --- 최소 AMR 스텁 --- #
class AMR:
    def __init__(self, aid: int, pos: Tuple[int, int]):
        self.id = aid
        self.pos = pos
        self.path: List[Tuple[int, int]] = []


def build_path_through_center(
    I: Intersection,
    start_pos: Tuple[int, int],
    exit_dir: str,
) -> List[Tuple[int, int]]:
    """
    register_amr가 current_arm / exit_arm을 제대로 잡을 수 있도록
    항상 center를 한 번 거쳐서 exit_dir 팔로 나가는 path를 만든다.
    """
    center = (I.center_x, I.center_y)
    if exit_dir in I.lane_coords and I.lane_coords[exit_dir]:
        exit_front = I.lane_coords[exit_dir][0]
    else:
        exit_front = start_pos

    if start_pos == center:
        return [center, exit_front]
    else:
        return [start_pos, center, exit_front]


def register(I: Intersection, amr: AMR, exit_dir: str):
    amr.path = build_path_through_center(I, amr.pos, exit_dir)
    I.register_amr(amr)


def snapshot_lanes(I: Intersection) -> Dict[str, List[Tuple[int, str] | None]]:
    """각 레인 셀에 (amr_id, exit_dir) 또는 None 저장"""
    lanes: Dict[str, List[Tuple[int, str] | None]] = {}
    pos2info: Dict[Tuple[int, int], Tuple[int, str]] = {}

    for aid, rec in I.amr_intent_map.items():
        amr = rec.get("amr_obj")
        if not amr:
            continue
        exit_dir = rec.get("exit_arm")
        pos2info[amr.pos] = (aid, exit_dir)

    for d, coords in I.lane_coords.items():
        lanes[d] = [pos2info.get(p) for p in coords]
    return lanes


def pprint_lanes(title: str, lanes: Dict[str, List[Tuple[int, str] | None]]):
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


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
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


def infer_arm(I: Intersection, pos: Optional[Tuple[int, int]]) -> Optional[str]:
    """좌표가 어느 팔/center에 있는지 arm label 반환 ('N','E','S','W','C' 또는 None)"""
    if pos is None:
        return None
    cx, cy = I.center_x, I.center_y
    if pos == (cx, cy):
        return "C"
    for d, coords in I.lane_coords.items():
        if pos in coords:
            return d
    return None


def final_arm_from_lanes(target_lanes: Dict[str, List[Optional[int]]]) -> Dict[int, str]:
    res: Dict[int, str] = {}
    for d, lane in target_lanes.items():
        for aid in lane:
            if aid is None:
                continue
            res[aid] = d
    return res


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
    input_dir_arg: Optional[str] = None,
    num_amrs_arg: Optional[int] = None,
):

    print(f"\n== Case #{case_idx}: random disperse_paths (seed={seed}) ==")
    if seed != 0:
        random.seed(seed)

    # 교차로 생성 (중앙 (10,10); 팔 길이는 인자)
    I = Intersection(
        intersection_data=(10, 10, lenN, lenE, lenS, lenW),
        present_dirs=None,
    )

    # 시작 위치 후보(레인 전체) 준비
    lane_cells: List[Tuple[int, int]] = []
    for d in I.dirs:
        lane_cells.extend(I.lane_coords[d])
    random.shuffle(lane_cells)

    # AMR 수 제한(레인 전체 칸 수 초과 방지)
    n_cap = min(n_amrs, len(lane_cells))
    chosen_starts = lane_cells[:n_cap]

    # 무작위 AMR 생성 & 등록(출구 방향도 랜덤)
    all_amrs: List[AMR] = []
    for idx, start in enumerate(chosen_starts, start=1):
        exit_dir = random.choice(list(I.dirs))
        amr = AMR(idx, start)
        register(I, amr, exit_dir)
        all_amrs.append(amr)

    # (선택) 어느 정도 확률로 center AMR 하나 만들어보기
    cx, cy = I.center_x, I.center_y
    center_xy = (cx, cy)
    if all_amrs and random.random() < 0.4:
        a = all_amrs[0]
        a.pos = center_xy
        register(I, a, random.choice(list(I.dirs)))  # center 기준으로 intent 다시 등록

    print("initial lanes:")
    pprint_lanes("lanes   ", snapshot_lanes(I))

    # 원래 arm / exit 정보
    original_arm: Dict[int, str] = {}
    exit_arm: Dict[int, str] = {}
    for aid, rec in I.amr_intent_map.items():
        original_arm[aid] = rec.get("current_arm")
        exit_arm[aid] = rec.get("exit_arm")

    all_ids = set(original_arm.keys())
    center_ids = {aid for aid, arm in original_arm.items() if arm == "C"}

    # input_dir, num_amrs 결정 (직접 지정 가능)
    #  - input_dir_arg 가 주어지면 그 값을 사용, 아니면 랜덤
    #  - num_amrs_arg 가 주어지면 min(num_amrs_arg, n_in_input), 아니면 랜덤
    if input_dir_arg is not None:
        if input_dir_arg not in I.dirs:
            raise ValueError(f"invalid --input-dir {input_dir_arg}, must be one of {I.dirs}")
        input_dir = input_dir_arg
    else:
        input_dir = random.choice(list(I.dirs))

    snap = snapshot_lanes(I)
    n_in_input = sum(1 for cell in snap.get(input_dir, []) if cell is not None)

    if num_amrs_arg is not None:
        # 실제 있는 개수보다 클 수 있으니 clamp
        num_amrs = min(num_amrs_arg, n_in_input)
    else:
        num_amrs = random.randint(0, n_in_input) if n_in_input > 0 else 0

    print(f"input_dir={input_dir}, num_amrs={num_amrs} (n_in_input={n_in_input})")

    # 실행
    target_lanes, paths = I.disperse_paths(input_dir=input_dir, num_amrs=num_amrs)

    # self.paths 가 반환값과 일치해야 함
    assert paths is I.paths or paths == I.paths

    # ----- 1) 기초 검증: path 형식 -----
    # 좌표 유효성
    ok, detail = all_coords_ok(I, paths)
    assert ok, f"⚠️ invalid coord found: {detail}"

    # 한 틱당 1칸 이동
    ok2, detail2 = steps_ok(paths)
    assert ok2, f"⚠️ 한 틱에 2칸 이상 이동: {detail2}"

    # path 길이 동기화 (남아 있는 AMR들에 대해)
    if paths:
        lens = {len(p) for p in paths.values()}
        assert len(lens) == 1, f"paths length mismatch: {lens}"

    # ----- 2) target_lanes와 최종 위치 일치 여부 -----
    final_pos: Dict[int, Tuple[int, int]] = {aid: p[-1] for aid, p in paths.items()}
    final_lanes_from_paths: Dict[str, List[Optional[int]]] = {d: [None] * len(I.lane_coords[d]) for d in I.dirs}
    for aid, pos in final_pos.items():
        for d, coords in I.lane_coords.items():
            if pos in coords:
                idx = coords.index(pos)
                assert final_lanes_from_paths[d][idx] is None, f"collision at {d}[{idx}]"
                final_lanes_from_paths[d][idx] = aid
                break

    for d in I.dirs:
        tgt_lane = target_lanes[d]
        fin_lane = final_lanes_from_paths[d]
        assert len(tgt_lane) == len(fin_lane)
        for i, (aid_tgt, aid_fin) in enumerate(zip(tgt_lane, fin_lane)):
            if aid_tgt is None:
                # paths에 없는 AMR일 수도 있으므로 fin_lane은 None일 수도 있고 아닐 수도 있다.
                continue
            if aid_tgt in paths:
                assert aid_fin == aid_tgt, f"{d}[{i}] expected {aid_tgt}, got {aid_fin}"

    # ----- 3) 팔 이동 제약 검증 -----
    #   - input_dir / center 출발이 아닌 AMR은 팔이 바뀌면 안 됨
    final_arm_map = final_arm_from_lanes(target_lanes)
    moved_from_input = 0

    for aid in all_ids:
        ia = original_arm[aid]
        fa = final_arm_map.get(aid)

        if ia == input_dir:
            if fa is not None and fa != ia:
                moved_from_input += 1
        elif ia == "C":
            # center 출발이면 어디로 가도 OK
            continue
        else:
            # 다른 팔에서 출발한 AMR은 팔이 바뀌면 안 됨
            if fa is not None:
                assert fa == ia, f"AMR {aid} moved from arm {ia} to {fa}, but only {input_dir}/center may move"

    assert moved_from_input <= num_amrs, (
        f"moved {moved_from_input} AMRs from {input_dir}, "
        f"but requested num_amrs={num_amrs}"
    )

    # ----- 4) 탈출 가능한 AMR + input_dir AMR 제거 로직 검증 -----
    alive_ids = set(paths.keys())
    removed_ids = all_ids - alive_ids

    expected_removed: set[int] = set()
    # 9번 블럭의 로직 그대로 재현
    for d in I.dirs:
        lane = target_lanes[d]

        for i in range(len(lane) - 1, -1, -1):  # far -> near
            aid = lane[i]
            if aid is None:
                continue

            cur_arm = original_arm[aid]
            ex_arm = exit_arm.get(aid)
            fin_arm = final_arm_map.get(aid)

            if d == input_dir:
                # input_dir 팔에 "남아 있는" AMR은 paths에서 제거
                expected_removed.add(aid)
                continue

            # (예외 1) 원래 center AMR
            if aid in center_ids:
                break

            # (예외 2) 다른 팔로 보내진 AMR (원래 팔과 최종 팔이 다르면)
            if fin_arm is not None and fin_arm != cur_arm:
                break

            # 여기까지 왔으면 "원래 팔에 그대로 남아 있는 AMR"
            if cur_arm == ex_arm:
                expected_removed.add(aid)
                continue
            else:
                break

    assert expected_removed == removed_ids, (
        f"탈출 AMR 제거 집합 불일치:\n"
        f"  expected={sorted(expected_removed)}\n"
        f"  actual  ={sorted(removed_ids)}"
    )

    # ----- 5) 시각화 및 요약 -----
    print_paths_tickwise(paths, center=center_xy, pad="repeat")

    if view and HAS_VIEWER:
        play_ticks_curses(I)

    print(f"alive_ids   = {sorted(alive_ids)}")
    print(f"removed_ids = {sorted(removed_ids)}")
    print("✅ disperse_paths case passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=1, help="몇 개의 랜덤 케이스를 돌릴지")
    ap.add_argument("--amrs", type=int, default=10, help="케이스당 AMR 수")
    ap.add_argument("--lenN", type=int, default=5)
    ap.add_argument("--lenE", type=int, default=5)
    ap.add_argument("--lenS", type=int, default=5)
    ap.add_argument("--lenW", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0, help="전역 시드(케이스별로 +i)")
    ap.add_argument("--no-view", action="store_true", help="tick_viewer 생략")

    # 새로 추가: input_dir, num_amrs 강제 지정
    ap.add_argument(
        "--input-dir",
        type=str,
        choices=["N", "E", "S", "W"],
        default="N",
        help="분산 테스트에 사용할 input_dir (N/E/S/W 중 하나). 미지정 시 랜덤."
    )
    ap.add_argument(
        "--num-amrs",
        type=int,
        default=3,
        help="disperse_paths에 전달할 num_amrs (미지정 시 랜덤)."
    )

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
            view=(not args.no_view),
            input_dir_arg=args.input_dir,
            num_amrs_arg=args.num_amrs,
        )

    print("\nAll disperse_paths random cases finished ✅")



if __name__ == "__main__":
    main()
