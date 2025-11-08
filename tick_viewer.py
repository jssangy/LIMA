# tick_viewer.py (업데이트)
import curses
from collections import defaultdict

def _label(aid: int) -> str:
    return f"{aid % 100:02d}"

def _pad_paths(paths: dict[int, list[tuple[int,int]]]):
    if not paths:
        return {}, 0
    T = max(len(p) for p in paths.values())
    padded = {}
    for aid, p in paths.items():
        padded[aid] = p + [p[-1]] * (T - len(p)) if p else [(0,0)] * T
    return padded, T

def _bbox(coords: set[tuple[int,int]]):
    xs = [x for x,_ in coords]
    ys = [y for _,y in coords]
    return min(xs), max(xs), min(ys), max(ys)

def play_ticks_curses(
    I,
    paths: dict[int, list[tuple[int,int]]] | None = None,
    cell_w: int = 3,
    # ▼ 숨김 옵션
    locked_ids: set[int] | None = None,      # 잠금된 AMR id들(있다면)
    drop_policy: str = "locked_then_center", # "none" | "on_first_center" | "on_stable_center" | "locked_only" | "locked_then_center"
):
    """
    Space/→: 다음 틱, b/←: 이전 틱, g: 이동, h: 숨김 토글, q: 종료

    drop_policy:
      - "none"                : 숨김 없음
      - "on_first_center"     : 첫 센터 도착 즉시 숨김
      - "on_stable_center"    : 남은 모든 프레임이 센터일 때부터 숨김
      - "locked_only"         : locked_ids에 한해 첫 센터 도착 즉시 숨김
      - "locked_then_center"  : locked_ids는 첫 센터에서 숨김 + 나머지는 on_stable_center
    """
    if paths is None:
        paths = getattr(I, "paths", {})
    padded, T = _pad_paths(paths)
    if T == 0:
        print("[viewer] paths 비어 있음. 먼저 actions_to_paths()를 실행하세요.")
        return

    center = (I.center_x, I.center_y)
    lane_cells = set(I.all_lane_coords) | {center}
    minx, maxx, miny, maxy = _bbox(lane_cells)

    # ---- 숨김 tick 계산 ----
    drop_at: dict[int, int] = {}
    def set_first_center(aid):
        P = padded.get(aid)
        if not P: return
        for t, pos in enumerate(P):
            if pos == center:
                drop_at[aid] = t
                break

    if drop_policy != "none":
        if drop_policy in ("locked_only", "locked_then_center") and locked_ids:
            for aid in locked_ids:
                set_first_center(aid)

        if drop_policy in ("on_first_center",):
            for aid in padded:
                if aid in drop_at: continue
                set_first_center(aid)

        if drop_policy in ("on_stable_center", "locked_then_center"):
            for aid, P in padded.items():
                if aid in drop_at:  # already decided by locked rule
                    continue
                # 가장 이른 t부터 P[t:]가 전부 center인 시점을 찾음
                for t in range(T):
                    if P[t] == center and all(pos == center for pos in P[t:]):
                        drop_at[aid] = t
                        break
    # 토글 가능
    hide_enabled = (drop_policy != "none")

    def render_frame(t: int):
        occ = defaultdict(list)
        hidden_now = []
        for aid, P in padded.items():
            # 숨김 조건 적용
            if hide_enabled and aid in drop_at and t >= drop_at[aid]:
                hidden_now.append(aid)
                continue
            occ[P[t]].append(aid)

        rows = []
        collisions = []
        for y in range(miny, maxy + 1):
            line = []
            for x in range(minx, maxx + 1):
                pos = (x, y)
                if pos in occ:
                    ids = occ[pos]
                    if len(ids) == 1:
                        s = _label(ids[0])
                    else:
                        s = "**" if cell_w >= 2 else "*"
                        collisions.append((pos, ids[:6]))
                else:
                    s = "C " if pos == center else ("· " if pos in lane_cells else "  ")
                if len(s) < cell_w: s += " " * (cell_w - len(s))
                elif len(s) > cell_w: s = s[:cell_w]
                line.append(s)
            rows.append("".join(line))
        return rows, collisions, hidden_now

    def main(stdscr):
        nonlocal hide_enabled  # ← 맨 앞에 선언!
        curses.curs_set(0)
        stdscr.nodelay(False)
        stdscr.keypad(True)
        t = 0

        while True:
            stdscr.clear()
            header = f"Tick {t}/{T-1}   [Space/→ next] [b/← prev] [g goto] [h hide:{'ON' if hide_enabled else 'OFF'}] [q quit]"
            stdscr.addstr(0, 0, header)

            rows, collisions, hidden_now = render_frame(t)
            for i, line in enumerate(rows, start=2):
                stdscr.addstr(i, 0, line)

            row_info = 3 + len(rows)
            if collisions:
                stdscr.addstr(row_info, 0, f"Collisions: {len(collisions)} ('**')")
                for j, (pos, ids) in enumerate(collisions[:5], start=1):
                    ids_str = ", ".join(_label(a) for a in ids)
                    stdscr.addstr(row_info + j, 0, f"  {pos}: {ids_str}")
                row_info += min(5, len(collisions)) + 1

            if hide_enabled:
                stdscr.addstr(row_info, 0, f"Hidden now: {len(hidden_now)}")
                row_info += 1

            stdscr.addstr(row_info, 0, "Label=id%100  ·=lane  C=center")

            key = stdscr.getch()
            if key in (ord('q'), 27):
                break
            elif key in (ord(' '), curses.KEY_RIGHT, ord('n')):
                t = min(t + 1, T - 1)
            elif key in (ord('b'), curses.KEY_LEFT, ord('p')):
                t = max(t - 1, 0)
            elif key == ord('g'):
                curses.echo()
                stdscr.addstr(row_info + 2, 0, f"Go to tick (0..{T-1}): ")
                try:
                    s = stdscr.getstr(row_info + 2, 22, 10).decode("utf-8")
                    tt = int(s.strip())
                    if 0 <= tt < T:
                        t = tt
                except Exception:
                    pass
                finally:
                    curses.noecho()
            elif key == ord('h'):
                hide_enabled = not hide_enabled  # ← nonlocal 선언 덕분에 재할당 OK

    curses.wrapper(main)
