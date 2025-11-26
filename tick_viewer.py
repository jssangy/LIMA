import curses
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set


def _label(aid: int, goal_dirs: Dict[int, str], id_width: int = 2) -> str:
    """라벨: '두 자리 ID + 방향문자'(예: 01N). 방향 없으면 '?'."""
    g = goal_dirs.get(aid, '?')
    return f"{aid % (10**id_years(id_width)):0{id_width}d}{g}"


def id_years(w: int) -> int:
    """내부용: 자리수 계산 헬퍼 (w자릿수 표현에 필요)."""
    return max(1, w)


def _pad_paths(paths: Dict[int, List[Tuple[int, int]]]) -> Tuple[Dict[int, List[Tuple[int, int]]], int]:
    """모든 AMR 경로 길이를 동일하게 패딩."""
    if not paths:
        return {}, 0
    max_len = max(len(p) for p in paths.values())
    out: Dict[int, List[Tuple[int, int]]] = {}
    for aid, p in paths.items():
        if not p:
            out[aid] = [(0, 0)] * max_len
        else:
            out[aid] = p + [p[-1]] * (max_len - len(p))
    return out, max_len


def _bbox(coords: Set[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    xs = [x for x, _ in coords]
    ys = [y for _, y in coords]
    return min(xs), max(xs), min(ys), max(ys)


def play_ticks_curses(
    I,
    paths: Optional[Dict[int, List[Tuple[int, int]]]] = None,
    *,
    cell_w: int = 4,              # 셀 폭(기본 4: '01N ' 정도)
    col_gap: int = 2,             # 열 간격(셀 사이 공백 수)
    row_gap: int = 2,             # 행 간격(빈 줄 수)
    id_width: int = 2,            # ID를 몇 자리로 찍을지 (기본 두 자리)
    show_empty: bool = False,     # 빈 칸(·/C) 표시 여부
    locked_ids: Optional[Set[int]] = None,
    drop_policy: str = "locked_then_center",  # 숨김 정책
):
    """
    Curses 기반 프레임 뷰어
    조작:
      Space/→ : 다음 틱
      b/←     : 이전 틱
      g       : 특정 틱으로 이동
      h       : 숨김 on/off 토글
      q/ESC   : 종료

    drop_policy:
      - "none"               : 숨김 없음
      - "on_first_center"    : 각 AMR이 처음 center에 도달한 시점 이후 숨김
      - "on_stable"          : 남은 프레임이 전부 center일 때부터 숨김
      - "locked_only"        : locked_ids에 포함된 AMR만 'on_first_center' 적용
      - "locked_then_center" : locked_ids는 on_first_center, 나머지는 on_stable

    **추가 동작**
      - 각 AMR은 자신의 원본 path 마지막 좌표까지 도달한 이후
        (즉, path 길이만큼 시간이 지난 뒤)부터는
        'Finished' 로 간주되어 시각화에서 사라진다.
    """
    # 1) 경로/프레임 수
    raw_paths = paths if paths is not None else getattr(I, "paths", {})
    padded, T = _pad_paths(raw_paths)
    if T == 0:
        print("[viewer] paths 비어 있음. 먼저 actions_to_paths()를 실행하세요.")
        return

    # ▶ 각 AMR별 완료 tick 계산
    #    - path 길이가 L이면, t = 0..L-1 동안만 실제 경로
    #    - t >= L 이면 '완료'로 보고 시각화에서 제거
    finish_at: Dict[int, int] = {}
    for aid, p in raw_paths.items():
        if not p:
            # 빈 경로면 t=0부터 이미 완료된 것으로 처리
            finish_at[aid] = 0
        else:
            finish_at[aid] = len(p)

    # 2) 공통 설정
    cell_w = max(cell_w, id_width + 1)                 # 'id+dir'을 위해 최소 폭 보장
    gap_str = " " * max(0, col_gap)

    center = (I.center_x, I.center_y)
    lane_cells = set(getattr(I, "all_lane_coords", set())) | {center}
    minx, maxx, miny, maxy = _bbox(lane_cells)

    # 3) 그릴 대상 ID 집합(= paths에 존재하는 AMR)
    ids_in_paths: Set[int] = set(padded.keys())

    # 4) 목표 방향 맵(출구 팔): amr_intent_map에서 읽어오되, paths에 있는 AMR만
    goal_dirs: Dict[int, str] = {}
    aimap = getattr(I, "abs_amr_intent_map", None) or getattr(I, "amr_intent_map", {})
    if isinstance(aimap, dict):
        for k, rec in aimap.items():
            try:
                aid = int(k)
            except Exception:
                continue
            if aid not in ids_in_paths:
                continue
            g = rec.get("exit_arm")
            if g in ("N", "E", "S", "W"):
                goal_dirs[aid] = g

    # 5) 숨김 정책 계산 (기존 로직 그대로)
    drop_at: Dict[int, int] = {}

    def mark_first_center(aid: int):
        P = padded.get(aid)
        if not P:
            return
        for tt, pos in enumerate(P):
            if pos == center:
                drop_at[aid] = tt
                break

    # locked set 기본값
    locked_ids = locked_ids or set()

    policy = drop_policy.lower()
    if policy != "none":
        if policy in ("locked_only", "locked_then_center"):
            for aid in locked_ids:
                mark_first_center(aid)
        if policy in ("on_first_center",):
            for aid in ids_in_paths:
                if aid not in drop_at:
                    mark_first_center(aid)
        if policy in ("on_stable", "locked_then_center"):
            for aid, P in padded.items():
                if aid in drop_at:
                    continue
                for tt in range(T):
                    if P[tt] == center and all(pp == center for pp in P[tt:]):
                        drop_at[aid] = tt
                        break

    hide_enabled = policy != "none"

    # 6) 프레임 렌더 함수
    def render_frame(t: int):
        occ: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        hidden_now: List[int] = []
        finished_now: List[int] = []

        for aid, P in padded.items():
            # ▶ 1순위: path 완료 여부
            f_t = finish_at.get(aid, T)
            if t >= f_t:
                finished_now.append(aid)
                continue

            # ▶ 2순위: drop_policy에 의한 숨김
            if hide_enabled and (aid in drop_at) and (t >= drop_at[aid]):
                hidden_now.append(aid)
                continue

            occ[P[t]].append(aid)

        rows: List[str] = []
        collisions: List[Tuple[Tuple[int, int], List[int]]] = []
        for y in range(miny, maxy + 1):
            line_cells: List[str] = []
            for x in range(minx, maxx + 1):
                pos = (x, y)
                if pos in occ:
                    ids = occ[pos]
                    if len(ids) == 1:
                        s = _label(ids[0], goal_dirs, id_width=id_width)
                    else:
                        s = "**" if cell_w >= 2 else "*"
                        collisions.append((pos, ids[:6]))
                else:
                    if show_empty:
                        s = "C " if pos == center else ("· " if pos in lane_cells else "  ")
                    else:
                        s = " " * cell_w
                # 고정폭 패딩
                if len(s) < cell_w:
                    s = s + " " * (cell_w - len(s))
                elif len(s) > cell_w:
                    s = s[:cell_w]
                line_cells.append(s)
            rows.append(gap_str.join(line_cells))
            for _ in range(max(0, row_gap)):
                rows.append("")  # 빈 줄 삽입
        return rows, collisions, hidden_now, finished_now

    # 7) Curses 메인 루프
    def main(stdscr):
        nonlocal hide_enabled
        curses.curs_set(0)
        stdscr.nodelay(False)
        stdscr.keypad(True)
        t = 0

        while True:
            stdscr.clear()
            header = (
                f"Tick {t}/{T-1}  "
                f"[Space/→ next] [b/← prev] [g goto] [h hide:{'ON' if hide_enabled else 'OFF'}] [q quit]  "
                f"(cell_w={cell_w}, col_gap={col_gap}, row_gap={row_gap}, empty={'ON' if show_empty else 'OFF'})"
            )
            try:
                stdscr.addstr(0, 0, header)
            except curses.error:
                pass  # 터미널 폭이 좁아도 무시

            rows, collisions, hidden_now, finished_now = render_frame(t)

            base_row = 2
            for i, line in enumerate(rows):
                try:
                    stdscr.addstr(base_row + i, 0, line)
                except curses.error:
                    # 화면이 좁을 때는 그릴 수 있는 만큼만 그린다
                    pass

            info_row = base_row + len(rows) + 1
            if collisions:
                try:
                    stdscr.addstr(info_row, 0, f"Collisions: {len(collisions)}  ('**')")
                    info_row += 1
                    for (pos, ids) in collisions[:5]:
                        labels = ", ".join(_label(a, goal_dirs, id_width) for a in ids)
                        stdscr.addstr(info_row, 0, f"  {pos}: {labels}")
                        info_row += 1
                except curses.error:
                    pass

            if hide_enabled:
                try:
                    stdscr.addstr(info_row, 0, f"Hidden now(by policy): {len(hidden_now)}")
                    info_row += 1
                except curses.error:
                    pass

            # ▶ 완료된 AMR 정보 표시
            if finished_now:
                try:
                    ids_str = ", ".join(str(a) for a in sorted(finished_now)[:10])
                    more = "" if len(finished_now) <= 10 else " ..."
                    stdscr.addstr(
                        info_row,
                        0,
                        f"Finished by now: {len(finished_now)}  (ids: {ids_str}{more})"
                    )
                    info_row += 1
                except curses.error:
                    pass

            try:
                stdscr.addstr(info_row, 0, "Label = id%100+dir   ·=lane   C=center")
            except curses.error:
                pass

            key = stdscr.getch()
            if key in (ord('q'), 27):  # 'q' 또는 ESC
                break
            elif key in (ord(' '), curses.KEY_RIGHT, ord('n')):
                t = min(t + 1, T - 1)
            elif key in (ord('b'), curses.KEY_LEFT):
                t = max(t - 1, 0)
            elif key == ord('g'):
                curses.echo()
                try:
                    stdscr.addstr(info_row + 2, 0, f"Go to tick (0..{T-1}): ")
                    s = stdscr.getstr(info_row + 2, 22, 10).decode("utf-8")
                    tt = int(s.strip())
                    if 0 <= tt < T:
                        t = tt
                except Exception:
                    pass
                finally:
                    curses.noecho()
            elif key == ord('h'):
                hide_enabled = not hide_enabled

    curses.wrapper(main)
