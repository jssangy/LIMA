import os
import json
import math
import numpy as np
from typing import Dict
from collections import defaultdict

from utils.AGV import agv
from utils.Intersection import Intersection
from utils import Funct
from utils.traffic_generator import TrafficGenerator, TrafficGenerator12, discover_border_arms_NxM
from utils.Controller import controller


class ENV():
    def __init__(self, prob_path):
        super().__init__()
        """환경 초기화"""
        base_dir = os.path.dirname(prob_path)
        with open(prob_path, 'r') as f:
            data = json.load(f)
        map_path = os.path.join(base_dir, data['mapFile'])
        
        self.map = self._load_map(map_path)
        self.walkable_tiles = np.count_nonzero(self.map == 0)
        print(f"Map loaded. Walkable tiles (value 0): {self.walkable_tiles}")
        self.intersection_data, self.full_adjacency = self._find_intersections_and_build_graph()
        if not self.intersection_data:
            raise ValueError("No intersection found in the map")
        
        self.time = 0
        self.agv_list = {}
        self.l_hop = 1
        self.max_steps = 10000

        # Intersections, Controller
        self.controller = controller(self.map)
        self.intersections: Dict[tuple, Intersection] = {
            f'x{inter_data[0]}y{inter_data[1]}': Intersection(inter_data, self.controller)
            for inter_data in self.intersection_data
        }
        self.deadlock_queue = []

        # TrafficGenerator
        arms12 = discover_border_arms_NxM(self.intersections)
        self.traffic_generator = TrafficGenerator12(arms12=arms12)
        self.traffic_generator.set_arm_gate(lambda iid, d: self.is_arm_outgoing_clear(iid, d))
        self.max_inside = 6

        # Color mapping
        self.color_map = Funct.Color_dict(6).dic
        self.prev_deadlock_map: dict[str, bool] = {}

        self.use_rl = False
        self.rl_policy = None

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
        self._spawn_amrs_if_needed()

        # 리셋 시에는 초기 관찰 상태만 반환
        obs, info = self.generate_observation()
        self.prev_deadlock_map: dict[str, bool] = {}

        return obs, info

    def step(self, actions=None, train=True):
        """
        actions: { "x{cx}y{cy}": action_idx, ... }
        반환: obs_next, reward_map, info_next
        """
        self.time += 1
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
                    moved.add(agv_id)
        
        # 나머지 AGV 이동
        for agv_id, agv_obj in self.agv_list.items():
            if agv_id not in moved:
                sig = self.controller.control_buffer.get(agv_id, (0, 0))
                if self._is_valid_move(agv_obj, sig):
                    agv_obj.move(sig)

        # 사용 후 정리
        self.controller.push_sequence = []

        # 4) 환경 변화 처리
        self._check_amr_completion()
        self._spawn_amrs_if_needed()

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

        # 2) 교차로 최신화 (여기서만 최신화! step에서는 하지 않음)
        self._update_intersections_state()

        # 3) 데드락 큐를 먼저 업데이트
        temp_info_for_queue = {
            iid: {"is_deadlock": I.is_deadlock} for iid, I in self.intersections.items()
        }
        self._update_deadlock_queue(temp_info_for_queue)

        # 4) 데드락/상태/마스크
        obs = {}
        info = {}

        for iid, I in self.intersections.items():
            state = np.asarray(I.get_state(), dtype=np.float32)
            edge_index = np.array([[0], [0]], dtype=np.int64)
            
            # [수정] deadlock_queue와 all_intersections를 인자로 전달
            action_mask = I.calculate_action_mask(self.deadlock_queue, self.intersections)
            action_mask = np.asarray(action_mask, dtype=np.bool_)

            obs[iid] = {
                "state": state,
                "edge_index": edge_index,
            }
            info[iid] = {
                "deadlock_active": I.center_deadlock,
                "is_deadlock": I.is_deadlock,
                "action_mask": action_mask,
            }
        
        return obs, info
    
    def _update_deadlock_queue(self, info):
        for iid, meta in info.items():
            if not isinstance(meta, dict):
                continue
            if bool(meta.get("is_deadlock", False)):
                if iid not in self.deadlock_queue:
                    self.deadlock_queue.append(iid)
            else:
                if iid in self.deadlock_queue:
                    self.deadlock_queue.remove(iid)

    def _inter_rank(self, iid):
        try:
            return self.deadlock_queue.index(iid)
        except ValueError:
            return math.inf

    def _is_valid_move(self, current_agv, control_signal):
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

    def _spawn_amrs_if_needed(self):
        # 새 TrafficGenerator12 규격 준수: 제너레이터가 지정한 arm에서만 스폰
        gen = getattr(self, "traffic_generator", None)
        if not gen or not gen.should_spawn_next():
            return

        tasks = gen.get_next_task_pair()
        if not tasks:
            return

        for t in tasks:
            agv_id = t["id"]
            start_iid = t["intersection_id"]           # 예: "x10y5"
            start_dir = t["start_direction"]           # "N"|"E"|"S"|"W"
            goal_iid  = t.get("goal_intersection_id", start_iid)
            goal_dir  = t.get("goal_direction", start_dir)

            start_pos = self._direction_to_coords(start_dir, start_iid)   # ← iid 사용
            goal_pos  = self._direction_to_coords(goal_dir, goal_iid)

            color = self.color_map.get(agv_id, (255, 0, 0))
            self.agv_list[agv_id] = agv(start_pos, agv_id, color)

            # 컨트롤러에 시작/목표 등록
            self.controller.add_agv(agv_id, start_pos, goal_pos)

    def _check_amr_completion(self):
        completed_agvs = []
        for agv_id, agv_obj in list(self.agv_list.items()):
            if agv_obj.pos == self.controller.agv_goal.get(agv_id):
                completed_agvs.append(agv_id)

        for agv_id in completed_agvs:
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
        kernel = np.array(
            [[1, 0, 1], 
             [0, 0, 0], 
             [1, 0, 1]])
        windows = np.lib.stride_tricks.sliding_window_view(self.map, kernel.shape)
        matches = np.all(windows == kernel, axis=(2, 3))
        centers = (np.argwhere(matches) + 1).tolist()
        return centers
        
    def _ray_len(self, r, c, dr, dc):
        H, W = self.map.shape
        length = 0
        rr, cc = r + dr, c + dc
        while 0 <= rr < H and 0 <= cc < W and self.map[rr][cc] == 0:
            if dr != 0:
                left_wall = (cc - 1 < 0) or (self.map[rr][cc - 1] == 1)
                right_wall = (cc + 1 >= W) or (self.map[rr][cc + 1] == 1)
                if not (left_wall or right_wall): break
            else:
                up_wall = (rr - 1 < 0) or (self.map[rr - 1][cc] == 1)
                down_wall = (rr + 1 >= H) or (self.map[rr + 1][cc] == 1)
                if not (up_wall or down_wall): break
            length += 1
            rr += dr
            cc += dc
        return length
    
    def _find_intersections_and_build_graph(self):
        """
        교차로 데이터 튜플을 ID로 사용하여 그래프를 생성하는 통합 함수.
        """
        intersections_data = []
        adj = defaultdict(list)
        
        # 1. 모든 교차로 중심 좌표를 찾음
        centers_rc = self._find_intersection_center()
        centers_xy = [(c, r) for r, c in centers_rc]
        
        # 2. 각 중심에 대해 intersection_data 튜플을 생성하고, 좌표-튜플ID 맵을 만듦
        center_xy_to_id = {}
        for c, r in centers_xy:
            len_N = self._ray_len(r, c, -1, 0)
            len_E = self._ray_len(r, c, 0, 1)
            len_S = self._ray_len(r, c, 1, 0)
            len_W = self._ray_len(r, c, 0, -1)

            if min(len_N, len_E, len_S, len_W) > 0:
                # intersection_data 튜플을 생성하여 ID로 사용
                current_id = (c, r, len_N, len_E, len_S, len_W)
                intersections_data.append(current_id)
                center_xy_to_id[(c, r)] = current_id

        # 3. 각 교차로에 대해 도로를 뻗어 연결성 검사
        for current_id in intersections_data:
            c, r, len_N, len_E, len_S, len_W = current_id
            
            # 각 방향의 도로 끝에서 한 칸 더 나아간 '연결 예상 좌표' 계산
            target_coords = {
                'N': (c, r - len_N - 1), 'E': (c + len_E + 1, r),
                'S': (c, r + len_S + 1), 'W': (c - len_W - 1, r)
            }

            # 4. 연결 예상 좌표가 다른 교차로의 중심과 일치하는지 확인
            for direction, target_coord in target_coords.items():
                if target_coord in center_xy_to_id:
                    neighbor_id = center_xy_to_id[target_coord]
                    # 양방향으로 연결 정보 추가
                    adj[current_id].append(neighbor_id)
                    adj[neighbor_id].append(current_id)
        
        # defaultdict를 일반 dict로 변환하고, 중복 제거 및 정렬
        final_adj = {k: sorted(list(set(v))) for k, v in adj.items()}

        return intersections_data, final_adj
    
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

    """
    def _center_occupied_any(self) -> bool:
        # (단일 교차로 가정) 교차로 중앙 점유 여부
        return self.intersection.center_agv is not None

    def _count_inside_intersection(self) -> int:
        # 교차로 내부(팔+중앙) AMR 수 (인덱스 사용)
        # agvs_in_intersection: set of AGV objects
        return len(self.intersection.agvs_in_intersection)

    def _arm_has_outgoing(self, direction: str) -> bool:
        # 해당 팔에서 바깥으로 나가려는(outgoing) AMR이 하나라도 있으면 True
        return bool(getattr(self.intersection, 'outgoing', {}).get(direction, False))

    def _spawn_gate(self, direction: str) -> bool:    
        # Poisson 스폰을 막는 글로벌 게이트:
        # - 중앙 점유 시 전체 스폰 정지
        # - 교차로 내부 AMR 수가 임계치 이상이면 정지
        # - 해당 팔 점유 시 해당 방향 스폰 금지

        if self._center_occupied_any():               # ★ 중앙 점유 금지
            return False
        if self._count_inside_intersection() >= self.max_inside:
            return False
        if self._arm_has_outgoing(direction):
            return False
        return True
    """

    # --- [GUI 연동을 위한 어댑터 함수들] ---
    def Get_AGV(self):
        """GUI가 AGV 목록을 가져갈 수 있도록 하는 함수"""
        return self.agv_list

    def get_active_tasks(self):
        """GUI가 AGV의 목표 지점을 가져갈 수 있도록 하는 함수"""
        return self.controller.agv_goal

    def make_info(self):
        # 스트리밍 모드: 타임리밋으로만 종료 신호
        if self.time >= self.max_steps:
            return False  # GUI에게 에피소드 종료 신호

        progress = self.traffic_generator.get_progress()
        # 바뀐 키들:
        # 'completed_pairs_in_cycle', 'total_pairs_per_cycle', 'completed_total', 'active_agvs', 'cycle'
        total_pairs_done = progress['completed_total']

        # 단순 스루풋(쌍/스텝). 필요하면 시간 스케일 맞춰 곱해 쓰세요.
        throughput = (total_pairs_done / self.time) if self.time > 0 else 0.0

        agv_states = {}
        for agv_id, agv_obj in self.agv_list.items():
            agv_states[agv_id] = [f"Goal_{agv_id}", 0]

        # GUI가 쓰던 포맷 유지: [완료수, 스루풋, AGV상태]
        return [total_pairs_done, throughput, agv_states]
