import os
import json
import math
import numpy as np
from typing import Dict
from collections import deque

from utils.AGV import agv
from utils.Intersection import Intersection
from utils import Funct
from utils.traffic_generator import TaskSetGenerator
from utils.Controller import controller


class ENV():
    def __init__(self, prob_path, max_arm_len_h=5, max_arm_len_v=5, num_amrs=500, max_steps=1000, running_opt=0):
        super().__init__()
        """환경 초기화"""
        base_dir = os.path.dirname(prob_path)
        with open(prob_path, 'r') as f:
            data = json.load(f)
        map_path = os.path.join(base_dir, data['mapFile'])

        self.time = 0
        
        self.map = self._load_map(map_path)
        self.walkable_tiles = np.count_nonzero(self.map == 0)
        print(f"Map loaded.")
        print(f"Map width: {self.map.shape[1]}, Map height: {self.map.shape[0]}")
        print(f"Walkable tiles (value 0): {self.walkable_tiles}")
        self.max_arm_len_h = max_arm_len_h
        self.max_arm_len_v = max_arm_len_v
        processed_intersections = self._find_intersections_and_build_graph()
        
        self.time = 0
        self.agv_list = {}
        self.l_hop = 1
        self.max_steps = max_steps

        self.controller = controller(self.map, running_opt=running_opt)

        self.intersections: Dict[str, Intersection] = {}
        for iid, inter_info in processed_intersections.items():
            self.intersections[iid] = Intersection(
                inter_info['data'], 
                self.controller, 
                inter_info['neighbors'],
                inter_info['present_dirs'],
            )

        self.deadlock_queue = []

        # TrafficGenerator
        self.traffic_generator = TaskSetGenerator(self.map, num_tasks=num_amrs)

        # Color mapping
        self.color_map = Funct.Color_dict(6).dic

        self.prev_deadlock_map: dict[str, bool] = {}

        self.use_rl = False
        self.rl_policy = None

        self.completed_agv_steps = []
        self.completed_agv_actions = [] # [신규] 완료된 AMR의 행동 카운트 저장 리스트

        self.completed_path_integrities: list[float] = []

        self._sig_hist = deque(maxlen=22)  # 전역 시그니처 히스토리
        self._stg_idle_win = 20             # 정지 판단 윈도우(최근 10스텝 모두 동일)
        self._stg_osc_win  = 20             # 진동 판단 윈도우(최근 10스텝이 ABABAB)
        self._stg_min_time = 20            # 초반 전이 구간 보호(20스텝 이전엔 감지 안 함)
        self._stg_enabled  = True          # 필요 시 끄고 켤 수 있음
        self._stg_reason   = None          # 디버그용(‘idle’/‘osc’)

    def reset(self):        
        self.time = 0
        self.tau_map: dict[str, int] = {}
        self.agv_list.clear()
        self.traffic_generator.start_new_episode()

        # 컨트롤러와 모든 교차로의 내부 상태 초기화
        self.controller.reset()
        for I in self.intersections.values():
            I.reset()
        self.deadlock_queue = []

        self._spawn_amrs_from_task_gen()

        # 리셋 시에는 초기 관찰 상태만 반환
        obs, info = self.generate_observation()
        self.prev_deadlock_map: dict[str, bool] = {}

        self.completed_agv_steps.clear()
        self.completed_agv_actions.clear()

        self.completed_path_integrities.clear()

        self._sig_hist.clear()
        self._stg_reason = None

        return obs, info

    def step(self, actions=None, train=True):
        """
        actions: { "x{cx}y{cy}": action_idx, ... }
        반환: obs_next, reward_map, info_next
        """
        # --- 에피소드 종료 조건 확인 ---
        terminated = False
        if self.traffic_generator.is_episode_done():
            terminated = True

        if self._update_and_check_stagnation():
            print(f"[EarlyStop] stagnation detected ({self._stg_reason}) at t={self.time}, "
                f"AGVs={len(self.agv_list)}")
            return False  # 테스트 루프에서 break
        
        # 공통 종료 조건: 최대 스텝 도달
        if self.time >= self.max_steps or terminated:
            return False
        
        if actions is None:
            actions = {}

        # 0) 현재 스냅샷
        obs_now, info_now = self.generate_observation()

        # 1) 액션 결정 (데드락인 동안 매 스텝 RL 보충)
        act_to_apply: dict[str, int] = dict(actions)

        # 1) 적용할 액션 결정 (RL 정책 보충)
        act_to_apply = dict(actions)
        if (not train) and self.use_rl and self.rl_policy:
            for iid, meta in info_now.items():
                # 유효한 교차로 정보인지 확인
                if not isinstance(meta, dict):
                    continue
                
                # 데드락이 활성화된 교차로에 대해서만 RL 정책 적용
                if meta.get("deadlock_active", False) and self.intersections[iid].center_agv and iid not in act_to_apply:
                    action_mask = meta.get("action_mask")
                    rl_action = int(self.rl_policy(obs_now[iid], action_mask))
                    act_to_apply[iid] = rl_action
        
        if self.use_rl and self.rl_policy:
            sorted_iids = sorted(self.intersections.keys(), key=self._inter_rank, reverse=True)

            # 2) 교차로별 플래닝 (begin → resolve → action → finalize)
            for I in self.intersections.values():
                I.begin_plan()

            for iid in sorted_iids:
                I = self.intersections[iid]

                # 사전 충돌 해결 로직 (예: 팔 스와핑)
                I.resolve_arm_swaps_all()

                # 액션 적용
                if iid in act_to_apply:
                    a_idx = act_to_apply[iid]
                    I.action_control(a_idx)

            final_plan_moves = {}
            final_plan_prio = {}
            final_plan_order = {}
            final_plan_owner = {}

            for iid in sorted_iids:
                I = self.intersections[iid]
                for agv_id, prio in I._plan_prio.items():
                    prev_prio = final_plan_prio.get(agv_id, -10**9)
                    if prio >= prev_prio:
                        final_plan_prio[agv_id] = prio
                        final_plan_moves[agv_id] = I._plan_moves[agv_id]
                        final_plan_order[agv_id] = I._plan_order[agv_id]
                        final_plan_owner[agv_id] = iid

            self.controller.control_buffer.update(final_plan_moves)
            
            items = []
            for agv_id, (prio, *order) in final_plan_order.items():
                owner_iid = final_plan_owner[agv_id]
                rank = self._inter_rank(owner_iid)
                items.append((rank, -prio, tuple(order), agv_id))
            items.sort()
            
            seq = []
            seen = set()
            for _, _, _, aid in items:
                if aid not in seen:
                    seen.add(aid)
                    seq.append(aid)
            self.controller.push_sequence = seq

            # 3) Movement 커밋 (컨트롤러의 우선순위 큐 먼저 처리)
            moved = set()
            for agv_id in self.controller.push_sequence:
                agv_obj = self.agv_list.get(agv_id)
                if agv_obj:
                    sig = self.controller.control_buffer.get(agv_id, (0, 0))
                    if self._is_valid_move(agv_obj, sig):
                        agv_obj.move(sig)
                        agv_obj.action_count += 1
                        moved.add(agv_id)
            
            # 나머지 AGV 이동
            for agv_id, agv_obj in self.agv_list.items():
                if agv_id not in moved:
                    sig = self.controller.control_buffer.get(agv_id, (0, 0))
                    if self._is_valid_move(agv_obj, sig):
                        agv_obj.move(sig)

            # 사용 후 정리
            self.controller.push_sequence = []
        
        else:
            for agv_id, agv_obj in self.agv_list.items():
                sig = self.controller.control_buffer.get(agv_id, (0, 0))
                if self._is_valid_move(agv_obj, sig):
                    agv_obj.move(sig)

        # 4) 환경 변화 처리
        self._check_amr_completion()

        self._spawn_amrs_from_task_gen()

        if self.controller.pibt_bump:
            for aid, inc in list(self.controller.pibt_bump.items()):
                if inc and aid in self.agv_list:
                    self.agv_list[aid].action_count += inc
            self.controller.pibt_bump.clear()

        # GUI/테스트 모드: 기존 요약 반환 유지
        if not train:
            return self.make_info()

        # 5) 다음 관측
        obs_next, info_next = self.generate_observation()

        # 6) 보상/이벤트 — 교차로별로 '개별 기록'
        reward_map: dict[str, float] = {}
        for iid, meta in info_next.items():
            if not isinstance(meta, dict):
                continue

            curr = bool(meta.get("deadlock_active", False))
            prev = self.prev_deadlock_map.get(iid, False)

            # 기본 보상 스킴(예시): 지속 -0.05, 해소 +1.0
            r = 0.0
            if curr:
                r -= 0.05
                self.tau_map[iid] = self.tau_map.get(iid, 0) + 1
            if prev and not curr:
                r += 1.0
                meta["tau"] = self.tau_map.get(iid, 0)
                self.tau_map[iid] = 0

            # 교차로별 이벤트 플래그/invalid_action 기록
            meta["event_start"] = (not prev) and curr
            meta["event_end"] = prev and (not curr)

            # prev 갱신 및 보상 저장
            self.prev_deadlock_map[iid] = curr
            reward_map[iid] = r

        # 종료/트렁케이트: 전역 상태만 간단 요약(합산 지표는 넣지 않음)
        terminated = False
        truncated = (self.time >= self.max_steps)
        info_next["_summary"] = {
            "terminated": terminated,
            "truncated": truncated,
            "time": self.time,
        }

        return obs_next, reward_map, info_next


    def generate_observation(self):
        """
        관측 생성(부작용 없음):
        - sensing/make_control
        - 교차로 최신화(_update_intersections_state)
        - 상태/마스크/푸시아웃 플래그 계산만 수행
        """
        # 1) 센싱/컨트롤 업데이트
        for agv_id, agv_obj in self.agv_list.items():
            self.controller.get_sensing(agv_id, agv_obj.pos)
        self.controller.make_control()
        if self.time == 0:
            # 리셋 직후: AGV별 초기 경로 설정
            for agv_id, paths in self.controller.agv_path.items():
                self.agv_list[agv_id].set_initial_path(paths)

        # 2) 교차로 최신화 (여기서만 최신화! step에서는 하지 않음)
        self._update_intersections_state()

        # 4) 데드락/상태/마스크
        obs = {}
        info = {}

        # [수정] _update_deadlock_queue를 먼저 호출하도록 순서 변경
        temp_info_for_queue = {
            iid: {"is_deadlock": I.is_deadlock} for iid, I in self.intersections.items()
        }
        self._update_deadlock_queue(temp_info_for_queue)

        for iid, I in self.intersections.items():
            state = np.asarray(I.get_state(), dtype=np.float32)
            
            # [수정] deadlock_queue만 인자로 전달
            action_mask = I.calculate_action_mask(self.deadlock_queue)
            action_mask = np.asarray(action_mask, dtype=np.bool_)

            obs[iid] = {
                "state": state,
            }
            info[iid] = {
                "deadlock_active": I.center_deadlock,
                "is_deadlock": I.is_deadlock,
                "action_mask": action_mask,
            }
        
        self.time += 1

        return obs, info
    
    def _update_deadlock_queue(self, info):
        """
        [수정] 데드락 큐를 새로운 우선순위 규칙에 따라 관리하고 정렬합니다.
        1순위: 교차로 내 AGV 수 (내림차순)
        2순위: 데드락 발생 시점 (오름차순)
        """
        queue_changed = False
        current_iids_in_queue = {item[0] for item in self.deadlock_queue}

        for iid, meta in info.items():
            if not isinstance(meta, dict):
                continue
            
            is_deadlocked = bool(meta.get("is_deadlock", False))

            if is_deadlocked:
                if iid not in current_iids_in_queue:
                    # 새로운 데드락 발생: (iid, 발생 시간) 추가
                    self.deadlock_queue.append((iid, self.time))
                    queue_changed = True
            else:
                if iid in current_iids_in_queue:
                    # 데드락 해소: 해당 iid 제거
                    self.deadlock_queue = [item for item in self.deadlock_queue if item[0] != iid]
                    queue_changed = True
        
        # 큐에 변화가 있을 때만 정렬 수행
        if queue_changed:
            # 정렬 키: (-AGV 수, 발생 시간)
            # AGV 수는 많을수록, 발생 시간은 이를수록 우선순위가 높다.
            self.deadlock_queue.sort(key=lambda item: (-len(self.intersections[item[0]].agvs_in_intersection), item[1]))

    def _inter_rank(self, iid):
        """
        [수정] 데드락 큐의 구조 변경에 맞춰 iid의 순위를 반환합니다.
        """
        try:
            # 큐는 (iid, timestamp) 튜플의 리스트이므로, iid만 추출하여 인덱스를 찾음
            iids_only = [item[0] for item in self.deadlock_queue]
            return iids_only.index(iid)
        except ValueError:
            return math.inf

    def _is_valid_move(self, current_agv, control_signal):

        if self.controller.running_opt in [3, 4]: # PIBT 충돌 시
            return True

        nx = current_agv.pos[0] + control_signal[0]
        ny = current_agv.pos[1] + control_signal[1]
        next_pos = (nx, ny)

        # 0) 경계/지형
        if not (0 <= nx < self.map.shape[1] and 0 <= ny < self.map.shape[0]):
            return False
        if self.map[ny][nx] == 1:
            return False
        
        # 2) 동일 칸 점유 충돌
        for other in self.agv_list.values():
            if other is not current_agv and next_pos == other.pos:
                return False
                
        # 3) 교차로 우선순위 규칙 (강화학습 활성화 시 적용)
        if self.rl_policy and self.use_rl:
            # --- 유틸: pos가 속한 교차로들(겹침 포함) ---
            def owners(pos):
                res = []
                for I in self.intersections.values():
                    if pos in I.all_lane_coords:   # all_lane_coords는 set이면 더 빠름
                        res.append(I)
                return res

            here_inters  = owners(current_agv.pos)
            target_inters = owners(next_pos)

            # 현재 위치 타입 분류
            cur_is_center  = any(current_agv.pos == (I.center_x, I.center_y) for I in here_inters)
            # 팔(arm)이라면 cur_is_center 둘 다 False

            # --- 우선순위 비교는 center에서만 수행 ---
            if cur_is_center:
                # 현재 위치의 랭크(겹침이면 최소 랭크), 없으면 inf
                cur_rank = min(
                    (self._inter_rank(getattr(I, "id", "")) for I in here_inters),
                    default=math.inf
                )

                # 다음 위치의 교차로(겹치면 현재 위치 교차로 제외)
                if here_inters:
                    here_ids = {getattr(I, "id", None) for I in here_inters}
                    candidates = [J for J in target_inters if getattr(J, "id", None) not in here_ids]
                else:
                    candidates = list(target_inters)

                # 다음 위치가 교차로가 아니면 우선순위 비교 스킵(허용)
                if candidates:
                    next_rank = min(self._inter_rank(getattr(J, "id", "")) for J in candidates)

                    # 네 규칙: next_rank < cur_rank -> 이동 불가
                    if next_rank < cur_rank:
                        return False
                # candidates가 비면 교차로가 아닌 칸으로 이동 → 비교 없이 통과

        return True
    
    def _update_intersections_state(self):
        for I in self.intersections.values():
            I.reset()

        for agv_id, agv_obj in self.agv_list.items():
            pos = agv_obj.pos
            for I in self.intersections.values():
                if pos in I.all_lane_coords:
                    I.add_agv(agv_obj)

        for I in self.intersections.values():
            I.check_deadlock()

    def _spawn_amrs_from_task_gen(self):
        """
        [이름 변경 및 Task 모드 전용]
        TaskSetGenerator로부터 새로운 AMR을 받아 환경에 추가.
        """
        gen = self.traffic_generator
        if not gen or not gen.should_spawn_next():
            return

        new_tasks = gen.get_next_task_pair(current_time=self.time)
        
        for task in new_tasks:
            agv_id = task['id']

            start_pos = tuple(task['start_pos'])
            goal_pos  = tuple(task['goal_pos'])

            new_agv = agv(start_pos, agv_id, self.color_map[agv_id % 6])
            self.agv_list[agv_id] = new_agv
            self.controller.add_agv(agv_id, start_pos, goal_pos)

    def _check_amr_completion(self):
        completed_agvs = []
        for agv_id, agv_obj in list(self.agv_list.items()):
            if agv_obj.pos == self.controller.agv_goal.get(agv_id):
                completed_agvs.append(agv_id)

        for agv_id in completed_agvs:
            agv_obj = self.agv_list[agv_id]
            if agv_obj is not None:
                pi_pct = agv_obj.path_integrity_ratio()
                self.completed_path_integrities.append(pi_pct)
                self.completed_agv_steps.append(agv_obj.steps)
                self.completed_agv_actions.append(agv_obj.action_count)
            self.traffic_generator.complete_task(agv_id)
            del self.agv_list[agv_id]
            self.controller.remove_agv(agv_id)

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
                elif c in ['.', 'E', 'S']: row.append(0)
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
                if not (left_wall or right_wall): break
            else:
                up_wall   = (rr - 1 < 0) or (self.map[rr - 1][cc] == 1)
                down_wall = (rr + 1 >= H) or (self.map[rr + 1][cc] == 1)
                if not (up_wall or down_wall): break

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
        
        # 1. 해당 팔에 나가는 AGV가 있으면 생성 금지
        has_outgoing = bool(getattr(I, "outgoing", {}).get(d, False))
        if has_outgoing:
            return False
            
        # 2. 해당 교차로가 데드락 상태이면 생성 금지
        is_deadlocked = iid in self.deadlock_queue
        if is_deadlocked:
            return False
            
        # 두 조건 모두 통과하면 생성 허용
        return True
    
    def _update_and_check_stagnation(self) -> bool:
        """
        최근 전역 위치 시그니처를 바탕으로 정지/진동을 감지.
        True면 조기 종료해야 함.
        """
        if not self._stg_enabled:
            return False
        if self.time < self._stg_min_time:
            self._sig_hist.clear()
            return False
        if not self.agv_list:
            self._sig_hist.clear()
            return False

        # 전역 시그니처: (agv_id, x, y) 튜플을 정렬한 튜플
        sig = tuple(sorted((aid, agv.pos[0], agv.pos[1]) for aid, agv in self.agv_list.items()))
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
            self._stg_reason = 'idle' if idle else 'osc'
            return True

        return False

    # --- [GUI 연동을 위한 어댑터 함수들] ---
    def Get_AGV(self):
        """GUI가 AGV 목록을 가져갈 수 있도록 하는 함수"""
        return self.agv_list

    def get_active_tasks(self):
        """GUI가 AGV의 목표 지점을 가져갈 수 있도록 하는 함수"""
        return self.controller.agv_goal

    def make_info(self):
        """
        [수정] GUI에 필요한 모든 정보를 계산하여 반환합니다.
        'task' 모드와 'traffic' 모드를 명시적으로 구분하여 처리합니다.
        """
        # --- 2. 모드에 따라 통계 정보 계산 ---
        progress = self.traffic_generator.get_progress()
        completed_tasks = progress.get('completed_total', 0)
        total_tasks = progress.get('spawned_total', 0)

        # Success Rate 계산
        success_rate = (completed_tasks / total_tasks) if total_tasks > 0 else 0.0
        
        # 스루풋 계산 (분 단위)
        throughput = (completed_tasks / self.time * 60) if self.time > 0 else 0.0

        # 평균 Action Count 계산 (완료 + 현재 활성 모두 포함)
        current_action_counts = [agv_obj.action_count for agv_obj in self.agv_list.values()]
        all_action_counts = list(self.completed_agv_actions) + current_action_counts
        avg_action_count = float(np.mean(all_action_counts)) if all_action_counts else 0.0

        active_pi = []
        for agv_obj in self.agv_list.values():
            active_pi.append(agv_obj.path_integrity_ratio())
        all_pi = self.completed_path_integrities + active_pi
        avg_pi = float(np.mean(all_pi)) if all_pi else 0.0

        # --- 3. 현재 활성화된 AGV들의 상세 정보 수집 ---
        active_agv_details = {}
        for agv_id, agv_obj in self.agv_list.items():
            active_agv_details[agv_id] = {
                "steps": agv_obj.steps,
                "action_count": agv_obj.action_count
            }

        # --- 4. 최종 정보 취합하여 반환 ---
        return {
            "success_rate": success_rate,
            "throughput": throughput,
            "avg_action_count": avg_action_count,
            "active_agvs": active_agv_details,
            "avg_path_integrity": avg_pi,
        }