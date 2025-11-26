import os
import json
import math
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from utils.AMR import AMR
from utils.Intersection import Intersection
from utils import Funct
from utils.traffic_generator import TaskSetGenerator, discover_border_arms_NxM, TrafficGenerator
from utils.Controller import AStarPlanner, PIBTPlanner, CBSPlanner, BFSPlanner


class ENV():
    def __init__(self, prob_path, max_arm_len_h=5, max_arm_len_v=5, num_amrs=500, max_steps=1024, running_opt=0, traffic_mode='task'):
        super().__init__()
        """환경 초기화"""
        base_dir = os.path.dirname(prob_path)
        with open(prob_path, 'r') as f:
            data = json.load(f)
        map_path = os.path.join(base_dir, data['mapFile'])
        self.goal = set()

        self.time = 0
        
        self.map = self._load_map(map_path)
        self.walkable_tiles = np.count_nonzero(self.map == 0)
        self.max_arm_len_h = max_arm_len_h
        self.max_arm_len_v = max_arm_len_v
        processed_intersections = self._find_intersections_and_build_graph()
        
        self.time = 0
        self.amr_list = {}
        self.max_steps = max_steps

        self.planner = BFSPlanner(self.map)

        self.intersections: Dict[str, Intersection] = {}
        for iid, inter_info in processed_intersections.items():
            self.intersections[iid] = Intersection(
                inter_info['data'],
                inter_info['present_dirs'],
            )
            
        # 교차로 간 이웃 맵핑 (양방향)
        self.iid_neighbors = {iid: set() for iid in processed_intersections.keys()}
        for iid, inter_info in processed_intersections.items():
            for nid in inter_info["neighbors"].values():
                if not nid:
                    continue
                # 한 방향
                self.iid_neighbors[iid].add(nid)
                # 반대 방향도 자동 연결
                if nid not in self.iid_neighbors:
                    self.iid_neighbors[nid] = set()
                self.iid_neighbors[nid].add(iid)

        # 각 셀이 어느 교차로에 속하는지 맵핑
        self.cell2iids: Dict[tuple[int, int], list[str]] = defaultdict(list)

        # 이벤트 셀들 (교차로 중심 + 레인 끝)
        self.event_cells = set()
        self.event_cells2iid = {}

        # 맵핑 구축
        for iid, I in self.intersections.items():
            center = (I.center_x, I.center_y)

            self.cell2iids[center].append(iid)
            self.event_cells.add(center)
            self.event_cells2iid[center] = iid

            for d in I.dirs:
                coords = I.lane_coords[d]
                for cell in coords:
                    self.cell2iids[cell].append(iid)

                end_cell = coords[-1]
                self.event_cells.add(end_cell)
                self.event_cells2iid[end_cell] = iid

        # 데드락 상태인 교차로
        self.deadlock_queue = []
        self.iid2sched: dict[str, set[int]] = defaultdict(set)

        self.traffic_mode = traffic_mode

        # TaskGenerator
        if self.traffic_mode == 'task':
            self.task_generator = TaskSetGenerator(self.map, num_tasks=num_amrs, goal_positions=self.goal)

        # Traffic Generator
        elif self.traffic_mode == 'traffic':
            arms = discover_border_arms_NxM(self.intersections)
            self.traffic_generator = TrafficGenerator(arms)
            self.traffic_generator.set_arm_gate(lambda iid, d: self.is_arm_outgoing_clear(iid, d))

        # Color mapping
        self.color_map = Funct.Color_dict(6).dic

        self.use_scheduler = False

        self.completed_amr_steps = []

        self.completed_path_integrities: list[float] = []

        self.time_ms = []


    def reset(self):        
        self.time = 0
        self.amr_list.clear()
        
        if self.traffic_mode == 'task':
            self.task_generator.start_new_episode()
        elif self.traffic_mode == 'traffic':
            self.traffic_generator.start_new_episode()

        # 모든 교차로의 내부 상태 초기화
        for I in self.intersections.values():
            I.reset()

        if self.traffic_mode == 'task':
            self._spawn_amrs_from_task_gen()
        elif self.traffic_mode == 'traffic':
            self._spawn_amrs_from_stream_gen()

        self.deadlock_queue = []
        self.iid2sched.clear()

        self.completed_amr_steps.clear()
        self.completed_path_integrities.clear()
        self.time_ms.clear()

        return

    
    def step(self):
        self.time += 1

        """        # 충돌 감지 (디버그용)
        final_positions = defaultdict(list)
        for amr_id, amr in self.amr_list.items():
            final_positions[amr.pos].append(amr_id)

        # 2개 이상 겹친 곳 필터링
        collisions = {pos: ids for pos, ids in final_positions.items() if len(ids) > 1}
        
        if collisions:
            error_msg = "\n[CRITICAL] Collision Detected at step {}:\n".format(self.time)
            
            for pos, ids in collisions.items():
                # 각 로봇의 ID와 스케줄링 여부를 문자열로 변환
                # 예: "10(Scheduled)", "5(Normal)"
                amr_details = []
                for aid in ids:
                    amr = self.amr_list[aid]
                    status = "Scheduled" if amr.scheduling > 0 else "Normal"
                    amr_details.append(f"{aid}({status})")
                
                error_msg += f"  Position {pos}: {', '.join(amr_details)}\n"
            
            raise RuntimeError(error_msg)
        """

        if self.traffic_mode == 'task' and self.task_generator.is_episode_done():
            return False

        # 1. 스케줄러 로직 (데드락 감지 및 해결)
        if self.use_scheduler:
            # (1) 교차로별 멤버 확인
            check_iids = set()
            iid2members: dict[str, list[int]] = defaultdict(list)

            for amr_id, amr_obj in self.amr_list.items():
                pos = tuple(amr_obj.pos)
                
                # 현재 위치가 어떤 교차로 영역에 속하는지 확인
                if pos in self.cell2iids:
                    for iid in self.cell2iids[pos]:
                        iid2members[iid].append(amr_id)
                        
                # 데드락 체크가 필요한지 확인
                if pos in self.event_cells:
                    iid = self.event_cells2iid[pos]
                    check_iids.add(iid)

            # (2) 잠금 해제 체크 (Deadlock 해제 시도)
            # 이미 잠긴 교차로 중, 스케줄링된 로봇들이 모두 빠져나갔다면 잠금 해제
            for iid in list(self.deadlock_queue):
                scheduled_members = self.iid2sched[iid]
                still_active = False

                for mid in list(scheduled_members):
                    amr = self.amr_list.get(mid, None)
                    if amr is None:
                        scheduled_members.discard(mid)
                        continue
                    if amr.scheduling > 0:
                        still_active = True
                    else:
                        scheduled_members.discard(mid)
                
                if not still_active:
                    self.deadlock_queue.remove(iid)

            # (3) 데드락 체크 및 스케줄링
            # 잠기지 않은 교차로에 대해서만 수행
            for iid in list(check_iids - set(self.deadlock_queue)):
                I = self.intersections[iid]
                I.reset()

                # 해당 교차로 영역에 있는 모든 AMR 등록
                for amr_id in iid2members[iid]:
                    amr_obj = self.amr_list[amr_id]
                    I.register_amr(amr_obj)

                if I.check_deadlock():
                    self.deadlock_queue.append(iid)
                    if iid not in self.deadlock_queue:
                        self.deadlock_queue.append(iid)
                    
                    # 1. 스케줄러 경로 생성
                    short_paths, target_exits = I.actions_to_paths()

                    for amr_id, short_path in short_paths.items():
                        if amr_id in self.amr_list:
                            amr_obj = self.amr_list[amr_id]
                            target_exit = target_exits[amr_id]
                            self.insert_scheduled_path(amr_obj, short_path, target_exit)
                            self.iid2sched[iid].add(amr_id)

                    paths_dict = {aid: amr.path for aid, amr in self.amr_list.items()}
                    self.print_paths_tickwise(paths_dict)

        # 2. AMR 이동
        # (A) 현재 위치 점유 맵 초기화
        current_occ = {amr.pos: amr.id for amr in self.amr_list.values()}

        # (B) 그룹 분리
        normal_amrs = [a for a in self.amr_list.values() if a.scheduling == 0]
        scheduled_amrs = [a for a in self.amr_list.values() if a.scheduling > 0]

        # -------------------------------------------------------
        # [Phase 1] 일반 로봇(Normal AMR) 먼저 이동
        # -------------------------------------------------------
        for amr in normal_amrs:
            next_pos = amr.next_pos
            if next_pos not in current_occ:
                if amr.pos in current_occ and current_occ[amr.pos] == amr.id:
                    del current_occ[amr.pos]
                amr.move()
                current_occ[amr.pos] = amr.id

        # -------------------------------------------------------
        # [Phase 2] 스케줄링 차단 여부 확인 (Blocking Check)
        # 일반 로봇들이 자리를 잡은 후, 스케줄링된 로봇들이 갈 수 있는지 확인
        # -------------------------------------------------------
        # 현재 일반 로봇들의 위치 집합
        normal_occ_pos = {amr.pos for amr in normal_amrs}

        # 차단된 교차로 ID 식별
        blocked_iids = set()

        for iid, members in self.iid2sched.items():
            for mid in members:
                amr = self.amr_list[mid]
                next_pos = amr.next_pos
                # 다음 위치가 일반 로봇에 의해 점유된 경우 차단
                if next_pos in normal_occ_pos:
                    blocked_iids.add(iid)
                    break

        
        # -------------------------------------------------------
        # [Phase 3] 스케줄링된 로봇 이동
        # -------------------------------------------------------
        for amr in scheduled_amrs:
            # 내가 속한 스케줄이 차단되었는지 확인
            is_blocked = False
            # 역추적: 내가 어느 iid에 속해있는지 확인 (iid2sched 순회)
            for iid, members in self.iid2sched.items():
                if amr.id in members:
                    if iid in blocked_iids:
                        is_blocked = True
                    break
            
            if is_blocked:
                # 그룹 전체가 대기해야 하므로 건너뜀
                continue

            # 이동 수행
            if amr.pos in current_occ and current_occ[amr.pos] == amr.id:
                del current_occ[amr.pos]
            amr.move()
            current_occ[amr.pos] = amr.id


        # 3. 완료 체크 및 정보 반환
        self._check_amr_completion()

        return self.make_info()


    def insert_scheduled_path(self, amr, short_path, target_exit):
        """
        교차로 스케줄러가 생성한 경로(short_path)를 현재 AMR 경로에 삽입한다.

        - short_path : amr.pos -> 교차로 내부 merge_point 까지의 스케줄 경로
        - target_exit: 이 AMR이 '원래' 나가고 싶어하는 출구 방향 lane의 마지막 셀(tip)

        최종 경로 구성:
        prefix(지금까지 온 경로) +
        short_path[1:] (현재 위치 이후 교차로 내부 스케줄) +
        bridge(merge_point → target_exit, BFS로 패치) +
        continuation(원래 AMR 경로에서 target_exit 이후 tail)
        """
        # 방어 코드
        if not short_path or len(short_path) < 2:
            return

        merge_point = short_path[-1]

        # --- 1) merge_point -> target_exit 까지 BFS (출구 tip까지) ---
        bridge = [merge_point]
        if target_exit is not None and target_exit != merge_point:
            try:
                # BFSPlanner 안의 실제 BFS 객체 사용
                bridge_path = self.planner.planner.plan_path(merge_point, target_exit)
            except AttributeError:
                # 혹시 BFSPlanner에 plan_path 메서드를 따로 추가했다면 이쪽을 쓰면 됨
                bridge_path = self.planner.plan_path(merge_point, target_exit)

            if bridge_path and len(bridge_path) >= 2:
                bridge = bridge_path
            else:
                print(f"[Scheduler] AMR {amr.id}: BFS bridge 실패 "
                    f"({merge_point} -> {target_exit}), merge_point만 사용.")
        else:
            # target_exit이 없거나 merge_point와 같으면 bridge는 [merge_point] 그대로
            pass

        # --- 2) 원래 amr 경로에서 target_exit 이후 tail 이어붙이기 ---
        continuation = []
        if target_exit is not None:
            exit_idx = -1
            # 현재 위치 이후 구간에서만 target_exit 검색
            for i in range(amr.path_cursor + 1, len(amr.path)):
                if amr.path[i] == target_exit:
                    exit_idx = i
                    break

            if exit_idx != -1 and exit_idx + 1 < len(amr.path):
                continuation = amr.path[exit_idx + 1:]
            elif exit_idx == -1:
                print(f"[Scheduler] AMR {amr.id}: target_exit {target_exit} "
                    f"not found in original path.")
            # exit_idx가 마지막 인덱스면 tail이 없으니 continuation은 빈 리스트 그대로

        # --- 3) 새 suffix 구성 ---
        new_suffix = []

        # (a) short_path: 현재 위치는 prefix에 있으니 [1:]부터
        new_suffix.extend(short_path[1:])

        # (b) bridge: merge_point 중복 방지를 위해 [1:]부터
        if len(bridge) > 1:
            new_suffix.extend(bridge[1:])

        # (c) target_exit 이후 원래 경로 tail
        new_suffix.extend(continuation)

        if not new_suffix:
            print(f"[Scheduler] AMR {amr.id}: new_suffix 비어 있음, 경로 변경 건너뜀.")
            return

        # --- 4) AMR 경로/상태 업데이트 ---
        prefix = amr.path[:amr.path_cursor + 1]

        amr.path = prefix + new_suffix

        # 스케줄된 구간 길이: short_path (중복 한칸 제거)
        sched_len = len(short_path) - 1
        amr.scheduling = sched_len

        # next_pos 동기화
        if len(amr.path) > amr.path_cursor + 1:
            amr.next_pos = amr.path[amr.path_cursor + 1]
        else:
            amr.next_pos = amr.pos



    
    def _fmt_xy(self, xy: Optional[Tuple[int, int]]) -> str:
        """(x,y) 튜플을 문자열로, None이면 빈 문자열."""
        if xy is None:
            return ""
        x, y = xy
        return f"({x},{y})"

    def print_paths_tickwise(self, paths: Dict[int, List[Tuple[int,int]]], *, pad: str = "repeat") -> None:
        """
        행 = tick, 열 = AMR ID 로 정렬해서 출력.

        paths : { amr_id: [(x,y), (x,y), ...] }
        pad   : "repeat" -> 각 AMR 경로의 마지막 좌표를 반복해서 패딩
                그 외 값 -> None 으로 패딩 (빈 칸으로 보임)
        """
        if not paths:
            print("\n[tickwise] (empty)")
            return

        # 열 순서: AMR id 오름차순
        amr_ids = sorted(paths.keys())
        # 전체 tick 수: 가장 긴 경로 길이
        max_len = max(len(p) for p in paths.values())

        # AMR별로 tick 길이 맞춰서 문자열 테이블 만들기
        table: Dict[int, List[str]] = {}
        for aid in amr_ids:
            p = paths[aid]
            if p and pad == "repeat":
                padded = p + [p[-1]] * (max_len - len(p))
            else:
                padded = p + [None] * (max_len - len(p))
            table[aid] = [self._fmt_xy(xy) for xy in padded]

        # 칸 폭 결정 (좌표/ID 중 가장 긴 것 기준)
        col_w = max(
            max(len(s) for aid in amr_ids for s in table[aid]),
            max(len(str(aid)) for aid in amr_ids),
        )

        print("\n[tickwise]")
        # 헤더: AMR id 가로로 쭉
        header = "tick ".ljust(6) + " ".join(f"{aid:>{col_w}}" for aid in amr_ids)
        print(header)
        print("-" * (6 + (col_w + 1) * len(amr_ids)))

        # 각 tick(T00, T01, ...) 한 줄씩 출력
        for t in range(max_len):
            row = " ".join(f"{table[aid][t]:>{col_w}}" for aid in amr_ids)
            print(f"T{t:02d}  {row}")




    def _spawn_amrs_from_task_gen(self):
        """
        [이름 변경 및 Task 모드 전용]
        TaskSetGenerator로부터 새로운 AMR을 받아 환경에 추가.
        """
        gen = self.task_generator
        if not gen or not gen.should_spawn_next():
            return

        new_tasks = gen.get_next_task_pair(current_time=self.time)
        
        for task in new_tasks:
            amr_id = task['id']

            start_pos = tuple(task['start_pos'])
            goal_pos = tuple(task['goal_pos'])

            new_amr = AMR(amr_id, start_pos, goal_pos, self.color_map[amr_id % 6])
            self.amr_list[amr_id] = new_amr
        
        self.planner.plan_for_new_amrs(self.amr_list)


    def _check_amr_completion(self):
        completed_amrs = []
        for amr_id, amr_obj in list(self.amr_list.items()):
            if amr_obj.pos == amr_obj.goal:
                completed_amrs.append(amr_id)

        for amr_id in completed_amrs:
            amr_obj = self.amr_list[amr_id]
            if amr_obj is not None:
                pi_pct = amr_obj.path_integrity_ratio()
                self.completed_path_integrities.append(pi_pct)
                self.completed_amr_steps.append(amr_obj.steps)
            if self.traffic_mode == 'task':
                self.task_generator.complete_task(amr_id)
            elif self.traffic_mode == 'traffic':
                self.traffic_generator.complete_task(amr_id)
            del self.amr_list[amr_id]


    def _spawn_amrs_from_stream_gen(self):
        """
        [새로 추가된 함수 - Traffic 모드 전용]
        TrafficGenerator로부터 새로운 AMR을 받아 환경에 추가.
        """
        gen = self.traffic_generator
        if not gen or not gen.should_spawn_next():
            return
        
        # TrafficGenerator12는 current_time 인자가 없음
        new_tasks = gen.get_next_task_pair()

        for task in new_tasks:
            amr_id = task['id']
            start_iid = task['intersection_id']
            start_dir = task['start_direction']
            goal_iid = task['goal_intersection_id']
            goal_dir = task['goal_direction']

            start_pos = self._direction_to_coords(start_dir, start_iid)
            goal_pos = self._direction_to_coords(goal_dir, goal_iid)

            if start_pos is None or goal_pos is None:
                continue
            
            # AMR 생성 및 등록 (amr 생성자 인자 순서 수정)
            new_amr = AMR(amr_id, start_pos, goal_pos, self.color_map[amr_id % 6])
            self.amr_list[amr_id] = new_amr
        
        self.planner.plan_for_new_amrs(self.amr_list)


    def _direction_to_coords(self, direction, intersection_ref):
        """
        direction: 'N'|'E'|'S'|'W'
        intersection_ref: 교차로 id 문자열("x{cx}y{cy}") 또는 (cx,cy,lenN,lenE,lenS,lenW) 튜플 모두 허용
        """
        # 1) iid 문자열 → Intersection에서 스펙 가져오기
        if isinstance(intersection_ref, str):
            I = self.intersections[intersection_ref]
            # outer_entry_cells 같은 사전이 있으면 그걸 우선 사용
            if hasattr(I, "outer_entry_cells") and direction in I.outer_entry_cells:
                return I.outer_entry_cells[direction]
            center_x, center_y = I.center_x, I.center_y
            len_N, len_E, len_S, len_W = I.len_N, I.len_E, I.len_S, I.len_W

        # 2) 과거 호환: 스펙 튜플로 온 경우
        else:
            center_x, center_y, len_N, len_E, len_S, len_W = intersection_ref

        direction_map = {
            'N': (center_x, center_y - len_N - 1),
            'E': (center_x + len_E + 1, center_y),
            'S': (center_x, center_y + len_S + 1),
            'W': (center_x - len_W - 1, center_y),
        }
        return direction_map[direction]

    
    def _load_map(self, map_path):        
        if not os.path.isfile(map_path): raise FileNotFoundError(f"Map file not found: {map_path}")
        map_data = []
        with open(map_path, 'r') as f: lines = f.readlines()
        map_start = None
        for idx, line in enumerate(lines):
            if line.strip() == 'map': map_start = idx + 1; break
        if map_start is None: raise ValueError("Map data not found in file")
        for line in lines[map_start:]:
            row = []
            for c in line.strip():
                if c in ['@', 'T']: row.append(1)
                elif c in ['.', 'E', 'S']: 
                    row.append(0)
                    if c == "S":
                        self.goal.add((len(row)-1, len(map_data)))  # (x,y)
                else: raise ValueError(f"Invalid character in map file: {c}")
            if row: map_data.append(row)
        return np.array(map_data)

    def _find_intersection_center(self):
        # 3x3 패턴들: 0=도로, 1=벽
        plus4 = np.array([
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1]
        ])

        # T자 (팔 하나 없는 방향)
        t_noN = np.array([  # 위쪽 팔 없음 (E/W/S만 열림)
            [1, 1, 1],
            [0, 0, 0],
            [1, 0, 1]
        ])
        t_noE = np.array([  # 오른쪽 팔 없음 (N/W/S만 열림)
            [1, 0, 1],
            [0, 0, 1],
            [1, 0, 1]
        ])
        t_noS = np.array([  # 아래쪽 팔 없음 (N/E/W만 열림)
            [1, 0, 1],
            [0, 0, 0],
            [1, 1, 1]
        ])
        t_noW = np.array([  # 왼쪽 팔 없음 (N/E/S만 열림)
            [1, 0, 1],
            [1, 0, 0],
            [1, 0, 1]
        ])

        kernels = (plus4, t_noN, t_noE, t_noS, t_noW)

        # 3x3 슬라이딩 윈도우
        windows = np.lib.stride_tricks.sliding_window_view(self.map, (3, 3))
        # 각 커널에 대해 매칭 후 OR 합치기
        match_any = np.zeros(windows.shape[:2], dtype=bool)
        for K in kernels:
            match_any |= np.all(windows == K, axis=(2, 3))

        # 윈도우 좌표 → 중심 좌표(슬라이딩 오프셋 +1)
        centers = (np.argwhere(match_any) + 1).tolist()
        return centers
        
    def _ray_len(self, r, c, dr, dc, max_len=None):
        H, W = self.map.shape
        length = 0
        rr, cc = r + dr, c + dc
        while 0 <= rr < H and 0 <= cc < W and self.map[rr][cc] == 0:
            if dr != 0:
                left_wall  = (cc - 1 < 0) or (self.map[rr][cc - 1] == 1)
                right_wall = (cc + 1 >= W) or (self.map[rr][cc + 1] == 1)
                if not (left_wall and right_wall): break
            else:
                up_wall   = (rr - 1 < 0) or (self.map[rr - 1][cc] == 1)
                down_wall = (rr + 1 >= H) or (self.map[rr + 1][cc] == 1)
                if not (up_wall and down_wall): break

            length += 1
            if max_len is not None and length >= max_len:
                break
            rr += dr
            cc += dc
        return length
    
    def _find_intersections_and_build_graph(self):
        centers_rc = self._find_intersection_center()
        centers_xy = [(c, r) for r, c in centers_rc]

        center_xy_to_data = {}
        for c, r in centers_xy:
            len_N = self._ray_len(r, c, -1, 0, max_len=self.max_arm_len_v)
            len_S = self._ray_len(r, c,  1, 0, max_len=self.max_arm_len_v)
            len_E = self._ray_len(r, c,  0, 1, max_len=self.max_arm_len_h)
            len_W = self._ray_len(r, c,  0,-1, max_len=self.max_arm_len_h)

            # ★ 사거리/삼거리 허용: 팔이 3개 이상 존재해야 교차로 인정
            present = {d for d, L in zip("NESW", [len_N, len_E, len_S, len_W]) if L > 0}
            if len(present) >= 3:
                center_xy_to_data[(c, r)] = (c, r, len_N, len_E, len_S, len_W, present)

        processed_intersections = {}
        for (c, r), tup in center_xy_to_data.items():
            c, r, len_N, len_E, len_S, len_W, present = tup
            current_iid = f'x{c}y{r}'

            # ★ 있는 팔만 이웃 계산
            neighbors_map = {}
            if 'N' in present:
                t = (c, r - len_N - 1)
                if t in center_xy_to_data:
                    nc, nr, *_ = center_xy_to_data[t]
                    neighbors_map['N'] = f'x{nc}y{nr}'
            if 'E' in present:
                t = (c + len_E + 1, r)
                if t in center_xy_to_data:
                    nc, nr, *_ = center_xy_to_data[t]
                    neighbors_map['E'] = f'x{nc}y{nr}'
            if 'S' in present:
                t = (c, r + len_S + 1)
                if t in center_xy_to_data:
                    nc, nr, *_ = center_xy_to_data[t]
                    neighbors_map['S'] = f'x{nc}y{nr}'
            if 'W' in present:
                t = (c - len_W - 1, r)
                if t in center_xy_to_data:
                    nc, nr, *_ = center_xy_to_data[t]
                    neighbors_map['W'] = f'x{nc}y{nr}'

            processed_intersections[current_iid] = {
                'data': (c, r, len_N, len_E, len_S, len_W),
                'neighbors': neighbors_map,
                # ↓ 이후 단계에서 마스크/상태 0패딩에 쓰기 좋게 전달
                'present_dirs': present,
            }
        return processed_intersections

    
    def is_arm_outgoing_clear(self, iid: str, d: str) -> bool:
        I = self.intersections[iid]

        # (선택) 삼거리 대응: 존재하지 않는 팔은 금지
        present = getattr(I, "present_dirs", set(I.lane_coords.keys()))
        if d not in present:
            return False

        # 1) 해당 팔에 '바깥으로 나가려는 흐름'이 있으면 금지
        has_outgoing = bool(getattr(I, "outgoing", {}).get(d, False))
        if has_outgoing:
            return False

        # 2) 교차로 데드락인 경우 금지  ← deadlock_queue 정규화
        dq = self.deadlock_queue or []
        if dq and isinstance(dq[0], tuple):
            dead_iids = {x for (x, _) in dq}
        else:
            dead_iids = set(dq)
        if iid in dead_iids:
            return False

        # (권장) 3) 팔 팁(outer entry)이 도로이고 비어있는지 확인
        if hasattr(I, "outer_entry_cells") and d in I.outer_entry_cells:
            tip = I.outer_entry_cells[d]
        else:
            cx, cy = I.center_x, I.center_y
            if   d == "N": tip = (cx, cy - I.len_N - 1)
            elif d == "E": tip = (cx + I.len_E + 1, cy)
            elif d == "S": tip = (cx, cy + I.len_S + 1)
            else:          tip = (cx - I.len_W - 1, cy)

        H, W = self.map.shape
        tx, ty = tip
        if not (0 <= tx < W and 0 <= ty < H):
            return False
        if self.map[ty][tx] == 1:
            return False
        if any(a.pos == tip for a in self.amr_list.values()):
            return False

        return True

    
    def _update_and_check_stagnation(self) -> bool:
        """
        최근 전역 위치 시그니처를 바탕으로 정지/진동을 감지.
        True면 조기 종료해야 함.
        """
        if self.time < self._stg_min_time:
            self._sig_hist.clear()
            return False
        if not self.amr_list:
            self._sig_hist.clear()
            return False

        # 전역 시그니처: (amr_id, x, y) 튜플을 정렬한 튜플
        sig = tuple(sorted((aid, amr.pos[0], amr.pos[1]) for aid, amr in self.amr_list.items()))
        self._sig_hist.append(sig)

        # 1) 정지: 최근 N개가 모두 동일
        idle = False
        if len(self._sig_hist) >= self._stg_idle_win:
            lastN = list(self._sig_hist)[-self._stg_idle_win:]
            idle = all(s == lastN[0] for s in lastN)

        # 2) 진동: 최근 M개가 ABABAB 형태(두 개의 시그니처가 번갈아)
        osc = False
        if len(self._sig_hist) >= self._stg_osc_win:
            w = self._stg_osc_win
            lastM = list(self._sig_hist)[-w:]
            if lastM[0] != lastM[1]:
                osc = all(lastM[i] == lastM[i % 2] for i in range(w))

        if idle or osc:
            return True

        return False
    

    # --- [GUI 연동을 위한 어댑터 함수들] ---
    def Get_AMR(self):
        """GUI가 AMR 목록을 가져갈 수 있도록 하는 함수"""
        return self.amr_list

    def get_active_tasks(self):
        """GUI가 AMR의 목표 지점을 가져갈 수 있도록 하는 함수"""
        return {amr_id: amr_obj.goal for amr_id, amr_obj in self.amr_list.items()}

    def make_info(self):
        """
        [수정] GUI에 필요한 모든 정보를 계산하여 반환합니다.
        'task' 모드와 'traffic' 모드를 명시적으로 구분하여 처리합니다.
        """
        # --- 2. 모드에 따라 통계 정보 계산 ---
        if self.traffic_mode == 'traffic':
            progress = self.traffic_generator.get_progress()
        elif self.traffic_mode == 'task':
            progress = self.task_generator.get_progress()
        completed_tasks = progress.get('completed_total', 0)
        total_tasks = progress.get('spawned_total', 0)

        # Success Rate 계산
        success_rate = (completed_tasks / total_tasks) if total_tasks > 0 else 0.0
        
        # 스루풋 계산 (분 단위)
        throughput = (completed_tasks / self.time * 60) if self.time > 0 else 0.0

        active_pi = []
        for amr_obj in self.amr_list.values():
            active_pi.append(amr_obj.path_integrity_ratio())
        all_pi = self.completed_path_integrities + active_pi
        avg_pi = float(np.mean(all_pi)) if all_pi else 0.0

        avg_ms = float(np.mean(self.time_ms)) if self.time_ms else 0.0

        # --- 3. 현재 활성화된 AMR들의 상세 정보 수집 ---
        active_amr_details = {}
        for amr_id, amr_obj in self.amr_list.items():
            active_amr_details[amr_id] = {
                "steps": amr_obj.steps,
            }

        # --- 4. 최종 정보 취합하여 반환 ---
        return {
            "success_rate": success_rate,
            "throughput": throughput,
            "active_amrs": active_amr_details,
            "avg_path_integrity": avg_pi,
            "avg_inference_time": avg_ms,
            "time": self.time,
        }
