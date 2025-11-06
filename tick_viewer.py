# tick_viewer.py
import curses
from collections import defaultdict

def _label(aid: int) -> str:
    # AMR 라벨: id의 하위 2자리 (고정 폭 2)
    return f"{aid % 100:02d}"

def _pad_paths(paths: dict[int, list[tuple[int,int]]]) -> tuple[dict[int, list[tuple[int,int]]], int]:
    if not paths:
        return {}, 0
    T = max(len(p) for p in paths.values())
    padded = {}
    for aid, p in paths.items():
        if not p:
            padded[aid] = [(0,0)] * T
            continue
        padded[aid] = p + [p[-1]] * (T - len(p))
    return padded, T

def _bbox(coords: set[tuple[int,int]]):
    xs = [x for x,_ in coords]
    ys = [y for _,y in coords]
    return min(xs), max(xs), min(ys), max(ys)

def play_ticks_curses(I, paths: dict[int, list[tuple[int,int]]] | None = None, cell_w: int = 3):
    """
    I: Intersection 인스턴스 (center/lanes 사용)
    paths: 생략 시 I.paths 사용 (actions_to_paths 실행 후)
    cell_w: 셀 폭(문자 수). 3~4 권장
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

    # 화면에 찍을 문자열 빌드
    def render_frame(t: int):
        # 점유 맵
        occ = defaultdict(list)
        for aid, p in padded.items():
            occ[p[t]].append(aid)

        rows = []
        collisions = []
        # 위쪽이 작은 y (화면 윗줄) — y 증가가 아래 방향
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
                        # 충돌 리스트 기록
                        collisions.append((pos, ids[:6]))  # 너무 길면 일부만
                else:
                    if pos == center:
                        s = "C "
                    elif pos in lane_cells:
                        s = "· "
                    else:
                        s = "  "
                # 고정 폭 셀
                if len(s) < cell_w:
                    s = s + " " * (cell_w - len(s))
                elif len(s) > cell_w:
                    s = s[:cell_w]
                line.append(s)
            rows.append("".join(line))
        return rows, collisions

    def main(stdscr):
        curses.curs_set(0)
        stdscr.nodelay(False)
        stdscr.keypad(True)
        t = 0

        while True:
            stdscr.clear()
            # 헤더
            header = f"Tick {t}/{T-1}   [Space/→ next] [b/← prev] [g goto] [q quit]"
            stdscr.addstr(0, 0, header)

            # 격자
            rows, collisions = render_frame(t)
            for i, line in enumerate(rows, start=2):
                stdscr.addstr(i, 0, line)

            # 충돌/라벨 보조정보
            row_info = 3 + len(rows)
            if collisions:
                stdscr.addstr(row_info, 0, f"Collisions: {len(collisions)} (표시는 '**')")
                for j, (pos, ids) in enumerate(collisions[:5], start=1):
                    ids_str = ", ".join(_label(a) for a in ids)
                    stdscr.addstr(row_info + j, 0, f"  {pos}: {ids_str}")
                row_info += min(5, len(collisions)) + 1

            # 라벨 안내
            stdscr.addstr(row_info, 0, "Label = id%100 (두 자리)  ·=lane  C=center")

            # 입력
            key = stdscr.getch()
            if key in (ord('q'), 27):  # q or ESC
                break
            elif key in (ord(' '), curses.KEY_RIGHT, ord('n')):
                t = min(t + 1, T - 1)
            elif key in (ord('b'), curses.KEY_LEFT, ord('p')):
                t = max(t - 1, 0)
            elif key == ord('g'):  # goto
                curses.echo()
                stdscr.addstr(row_info + 2, 0, "Go to tick (0..{}): ".format(T - 1))
                try:
                    s = stdscr.getstr(row_info + 2, 22, 10).decode("utf-8")
                    tt = int(s.strip())
                    if 0 <= tt < T:
                        t = tt
                except Exception:
                    pass
                finally:
                    curses.noecho()

    curses.wrapper(main)


# ---- 데모 ----
if __name__ == "__main__":
    # 데모: I.paths 가 있다고 가정하지 않으므로 간단한 가짜 paths 생성
    class DummyI:
        center_x = 10; center_y = 10
        # 십자 교차로 3칸
        lane_coords = {
            'N': [(10,9),(10,8),(10,7)],
            'E': [(11,10),(12,10),(13,10)],
            'S': [(10,11),(10,12),(10,13)],
            'W': [(9,10),(8,10),(7,10)]
        }
        present_dirs = {'N','E','S','W'}
        all_lane_coords = set(sum(lane_coords.values(), [])) | {(center_x, center_y)}
        # 예시 paths
        paths = {
            1: [(10,7),(10,8),(10,9),(10,10),(11,10),(11,10)],
            2: [(12,10),(11,10),(10,10),(10,9),(10,9),(10,10)],
            3: [(10,11),(10,11),(10,10),(9,10),(9,10),(9,10)],
        }

    I = DummyI()
    play_ticks_curses(I)  # space로 넘겨보세요
