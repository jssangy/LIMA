import random
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable

DIRS = ("N", "E", "S", "W")

class TrafficGenerator:
    """
    다방향 스트리밍 트래픽 생성기.
    - 매 스텝 방향별 독립 Poisson(λ_d)로 유입을 샘플.
    - 한 스텝에 최대 K개까지만 스폰(라운드로빈), 초과분은 방향별 backlog에 적재.
    - 출구(목표)는 p_turn(d->d')에 따라 샘플(기본: 균등, 자기 방향 제외).
    - ENV와의 호환을 위해:
        - should_spawn_next() : 항상 True
        - get_next_task_pair(): 이번 스텝에 스폰할 작업 리스트 반환(0~K개)
        - complete_task(agv_id) / get_progress() 제공
        - is_episode_done(): 스트리밍이라 항상 False
    """
    def __init__(
        self,
        active_dirs: Optional[List[str]] = None,     # 활성 방향 집합
        lam: float = 0.1,                           # 기본 λ (per-step, per-dir)
        lambda_per_dir: Optional[Dict[str, float]] = None,
        max_spawn_per_step: int = 4,                 # 한 스텝 최대 스폰 수 (K)
        turn_probs: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.rng = np.random.default_rng()
        self.active_dirs = tuple(active_dirs) if active_dirs else DIRS
        self.lambda_d = {d: (lambda_per_dir[d] if lambda_per_dir and d in lambda_per_dir else lam)
                         for d in self.active_dirs}
        self.max_spawn_per_step = max(1, int(max_spawn_per_step))
        self.turn_probs = self._build_turn_probs(turn_probs)
        self.agv_id_counter = 0

        # 방향별 backlog(슬롯이 안 나면 대기)
        self.backlog = {d: 0 for d in self.active_dirs}

        # 통계
        self.spawned_total = 0
        self.completed_total = 0
        self.step_count = 0

        # 호환성(예전 Color_dict 초기화 등에서 참조)
        self.total_tasks_in_episode = len(self.active_dirs)  # 의미적으론 무관

        # 용량 게이트(선택): set_capacity_gate(fn)으로 설정하면
        # fn(dir:str) -> bool 일 때만 스폰
        self._capacity_gate = None

    # ---------- 외부 인터페이스 ----------
    def start_new_episode(self, reset_ids: bool = True):
        """에피소드 시작(스트리밍). 기본적으로 ID/통계 리셋."""
        if reset_ids:
            self.agv_id_counter = 0
            self.spawned_total = 0
            self.completed_total = 0
            self.step_count = 0
        self.backlog = {d: 0 for d in self.active_dirs}

    def should_spawn_next(self) -> bool:
        """스트리밍: 매 스텝 스폰 시도."""
        return True

    def get_next_task_pair(self) -> List[Dict]:
        """
        이번 스텝에 실제 스폰할 작업 리스트(0~K개) 반환.
        - 백로그 없음
        - 각 활성 방향 d에 대해 arrivals ~ Poisson(λ_d) 샘플 → arrivals>0 이면 1개 스폰 시도
        - 용량 게이트(_capacity_gate)가 있으면 True일 때만 스폰
        - 한 스텝 총 스폰 수는 max_spawn_per_step(K)로 상한
        """
        self.step_count += 1
        tasks: List[Dict] = []

        # 방향 편향 방지
        dirs_rr = list(self.active_dirs)
        random.shuffle(dirs_rr)

        for d in dirs_rr:
            if len(tasks) >= self.max_spawn_per_step:
                break

            arrivals = int(self.rng.poisson(self.lambda_d[d]))
            if arrivals <= 0:
                continue  # 이번 스텝엔 이 방향에서 스폰 없음

            # ENV에서 설정한 게이트(예: 중앙 점유, 해당 방향 egress 존재 등)
            if self._capacity_gate is not None and not self._capacity_gate(d):
                continue

            g = self._sample_goal(d)
            tasks.append({
                "id": self.agv_id_counter,
                "start_direction": d,
                "goal_direction": g,
            })
            self.agv_id_counter += 1
            self.spawned_total += 1

        return tasks


    def complete_task(self, agv_id: int):
        """ENV에서 완료시 호출(통계용)."""
        self.completed_total += 1

    def is_episode_done(self) -> bool:
        """스트리밍 모드: 종료 없음(타임리밋은 ENV에서 처리)."""
        return False

    def get_progress(self) -> Dict:
        """로깅/GUI용 진행 상황."""
        return {
            "spawned_total": self.spawned_total,
            "completed_total": self.completed_total,
            "backlog": dict(self.backlog),
            "active_dirs": self.active_dirs,
            "lambdas": self.lambda_d,
            "step": self.step_count,
        }

    # ---------- 선택: 용량 게이트 연결 ----------
    def set_capacity_gate(self, fn):
        """
        L=1 등에서 슬롯이 비어있을 때만 스폰하려면 ENV에서
        generator.set_capacity_gate(lambda d: slot_free(d)) 로 연결.
        """
        self._capacity_gate = fn

    # ---------- 내부 유틸 ----------
    def _build_turn_probs(self, turn_probs):
        # 기본: 자기 방향 제외 3방향 균등
        base = {}
        for d in DIRS:
            others = [x for x in DIRS if x != d]
            p = {x: 1/3 for x in others}
            base[d] = p
        if not turn_probs:
            return base
        # 사용자 지정이 있으면 병합(정규화)
        merged = {}
        for d in DIRS:
            p = dict(base[d])
            if d in turn_probs:
                p.update(turn_probs[d])
            # 자기 방향 제거 + 정규화
            p = {k: v for k, v in p.items() if k != d}
            s = sum(p.values())
            merged[d] = {k: (v / s if s > 0 else 1/3) for k, v in p.items()}
        return merged

    def _sample_goal(self, start_dir: str) -> str:
        probs = self.turn_probs[start_dir]
        ks, vs = zip(*probs.items())
        return self.rng.choice(ks, p=np.array(vs, dtype=float))


DIRS = ("N", "E", "S", "W")

def discover_border_arms_NxM(intersections: Dict[str, object]) -> List[Tuple[str, str]]:
    """
    [새로 추가된 함수]
    N x M 그리드에서 맵 바깥을 향하는 모든 외곽 팔을 찾아 반환.
    """
    if not intersections:
        return []

    # 1) 모든 교차로의 x, y 좌표를 수집하여 고유한 값만 정렬
    all_x = sorted(list({I.center_x for I in intersections.values()}))
    all_y = sorted(list({I.center_y for I in intersections.values()}))

    N = len(all_x)
    M = len(all_y)

    if N == 0 or M == 0:
        return []

    # 2) 좌표 값을 그리드 인덱스(0, 1, ...)로 변환하기 위한 맵 생성
    x_to_idx = {x: i for i, x in enumerate(all_x)}
    y_to_idx = {y: i for i, y in enumerate(all_y)}

    # 3) 외곽 팔 수집
    border_arms: List[Tuple[str, str]] = []
    for iid, I in intersections.items():
        ix = x_to_idx.get(I.center_x)
        iy = y_to_idx.get(I.center_y)

        if ix is None or iy is None:
            continue

        # 가장 왼쪽 열에 있는 교차로의 '서쪽(W)' 팔
        if ix == 0:
            border_arms.append((iid, "W"))
        # 가장 오른쪽 열에 있는 교차로의 '동쪽(E)' 팔
        if ix == N - 1:
            border_arms.append((iid, "E"))
        # 가장 위쪽 행에 있는 교차로의 '북쪽(N)' 팔
        if iy == 0:
            border_arms.append((iid, "N"))
        # 가장 아래쪽 행에 있는 교차로의 '남쪽(S)' 팔
        if iy == M - 1:
            border_arms.append((iid, "S"))
            
    print(f"Discovered {len(border_arms)} border arms from a {N}x{M} grid.")
    return border_arms


class TrafficGenerator12:
    """
    외곽 팔에서만 스폰하는 트래픽 생성기.
    - 각 arm별로 Poisson 분포에 따라 AGV 생성.
    - [수정] goal은 항상 출발 팔의 '반대 방향'에 있는 팔 중에서 균등 샘플.
    - arm_gate 콜백이 False면 해당 팔에서 생성하지 않음.
    - max_agvs를 초과하여 생성하지 않음.
    """
    def __init__(
        self,
        arms12: List[Tuple[str, str]],
        lam: float = 0.05,
        lambda_per_arm: Optional[Dict[Tuple[str, str], float]] = None,
        arm_gate: Optional[Callable[[str, str], bool]] = None,
        debug: bool = False,
        max_agvs: int = 500,
    ):
        self.rng = np.random.default_rng()
        self.arms = [(str(iid), d) for (iid, d) in arms12]
        self.lam = float(lam)
        self.lambda_arm = {
            (iid, d): (lambda_per_arm.get((iid, d), self.lam) if lambda_per_arm else self.lam)
            for (iid, d) in self.arms
        }
        self.agv_id_counter = 0
        self.spawned_total = 0
        self.completed_total = 0
        self.step_count = 0
        self.max_agvs = max_agvs

        # [추가] 방향별로 팔을 분류하고, 반대 방향을 매핑
        self.arms_by_direction = {"N": [], "S": [], "E": [], "W": []}
        for iid, d in self.arms:
            if d in self.arms_by_direction:
                self.arms_by_direction[d].append((iid, d))
        
        self.opposite_direction = {"N": "S", "S": "N", "E": "W", "W": "E"}

        # [추가] 유효성 검사: 모든 방향에 대해 반대 방향 팔이 존재하는지 확인
        for direction, arm_list in self.arms_by_direction.items():
            if arm_list:  # 해당 방향에 출발 팔이 있다면
                opposite_dir = self.opposite_direction[direction]
                if not self.arms_by_direction[opposite_dir]:
                    print(f"Warning: Arms exist for direction '{direction}', but no arms found for the opposite direction '{opposite_dir}'. This may cause errors.")

        self._arm_gate = arm_gate
        self.debug = bool(debug)

    def set_arm_gate(self, fn: Callable[[str, str], bool]) -> None:
        self._arm_gate = fn

    # --- 외부 인터페이스 ---
    def start_new_episode(self, reset_ids: bool = True):
        if reset_ids:
            self.agv_id_counter = 0
            self.spawned_total = 0
            self.completed_total = 0
            self.step_count = 0

    def should_spawn_next(self) -> bool:
        return True

    def get_next_task_pair(self) -> List[Dict]:
        self.step_count += 1
        tasks: List[Dict] = []
        active_agvs = self.spawned_total - self.completed_total

        if active_agvs >= self.max_agvs:
            return []

        arms_rr = list(self.arms)
        random.shuffle(arms_rr)
        for (iid, d) in arms_rr:
            if (active_agvs + len(tasks)) >= self.max_agvs:
                break

            if self._arm_gate is not None and not self._arm_gate(iid, d):
                continue

            k = int(self.rng.poisson(self.lambda_arm.get((iid, d), self.lam)))
            if k <= 0:
                continue
            
            for _ in range(k):
                if (active_agvs + len(tasks)) >= self.max_agvs:
                    break

                # [수정] 새로운 목적지 샘플링 함수 호출
                try:
                    gid, gd = self._sample_goal_opposite((iid, d))
                    tasks.append({
                        "id": self.agv_id_counter,
                        "intersection_id": iid,
                        "start_direction": d,
                        "goal_intersection_id": gid,
                        "goal_direction": gd,
                    })
                    self.agv_id_counter += 1
                    self.spawned_total += 1
                except ValueError as e:
                    if self.debug:
                        print(f"[TG12:{self.step_count}] Goal sampling error: {e}")
                    continue # 목적지를 찾을 수 없으면 해당 AGV는 생성하지 않음

        return tasks

    def complete_task(self, agv_id: int):
        self.completed_total += 1

    def is_episode_done(self) -> bool:
        return False

    def get_progress(self) -> Dict:
        return {
            "spawned_total": self.spawned_total,
            "completed_total": self.completed_total,
            "active_agvs": self.spawned_total - self.completed_total,
            "max_agvs": self.max_agvs,
            "step": self.step_count,
            "arms": list(self.arms),
            "lambdas": {f"{iid}:{d}": self.lambda_arm.get((iid,d)) for (iid, d) in self.arms},
        }

    # --- 내부 ---
    def _sample_goal_opposite(self, src_arm: Tuple[str, str]) -> Tuple[str, str]:
        """
        [새로운 함수]
        출발 팔의 반대 방향에 있는 팔들 중에서 목적지를 균등하게 샘플링.
        """
        _, src_dir = src_arm
        
        target_dir = self.opposite_direction[src_dir]
        candidate_goals = self.arms_by_direction[target_dir]
        
        if not candidate_goals:
            raise ValueError(f"No goal arms found for opposite direction '{target_dir}'.")

        goal_idx = self.rng.integers(len(candidate_goals))
        return candidate_goals[goal_idx]
    

class TaskSetGenerator:
    """
    [새로 추가된 클래스]
    에피소드 시작 시, 모든 외곽 팔에 AGV를 하나씩 배치하고,
    서로 겹치지 않는 반대편 목적지를 할당하는 고정된 Task Set을 생성.
    """
    def __init__(self, all_arms: List[Tuple[str, str]], seed: Optional[int] = 7):
        self.rng = np.random.default_rng(seed)
        self.all_arms = all_arms
        self.agv_id_counter = 0
        self.task_set = []
        self.tasks_dispatched = True # 에피소드 시작 시 False로 설정됨

        # 방향별로 팔 분류
        self.arms_by_direction = {"N": [], "S": [], "E": [], "W": []}
        for iid, d in self.all_arms:
            if d in self.arms_by_direction:
                self.arms_by_direction[d].append((iid, d))
        
        self.opposite_direction = {"N": "S", "S": "N", "E": "W", "W": "E"}

    def start_new_episode(self, reset_ids: bool = True):
        """에피소드 시작 시 고정된 Task Set을 생성."""
        if reset_ids:
            self.agv_id_counter = 0
        
        self.task_set = []
        
        # N <-> S, E <-> W 그룹에 대해 1:1 매칭 수행
        self._create_one_to_one_tasks("N", "S")
        self._create_one_to_one_tasks("S", "N")
        self._create_one_to_one_tasks("E", "W")
        self._create_one_to_one_tasks("W", "E")

        self.tasks_dispatched = False
        print(f"Generated a fixed task set with {len(self.task_set)} AGVs.")

    def _create_one_to_one_tasks(self, start_dir: str, goal_dir: str):
        """두 방향 그룹 간에 겹치지 않는 Task를 생성."""
        start_arms = self.arms_by_direction[start_dir][:]
        goal_arms = self.arms_by_direction[goal_dir][:]

        # 겹치지 않는 매칭을 위해 양쪽 리스트를 섞음
        random.shuffle(start_arms)
        random.shuffle(goal_arms)

        # 두 그룹 중 더 작은 쪽의 크기만큼 Task 생성
        num_tasks = min(len(start_arms), len(goal_arms))

        for i in range(num_tasks):
            start_iid, start_d = start_arms[i]
            goal_iid, goal_d = goal_arms[i]
            
            self.task_set.append({
                "id": self.agv_id_counter,
                "intersection_id": start_iid,
                "start_direction": start_d,
                "goal_intersection_id": goal_iid,
                "goal_direction": goal_d,
            })
            self.agv_id_counter += 1

    def get_next_task_pair(self) -> List[Dict]:
        """처음 호출 시 준비된 Task Set 전체를 반환하고, 이후에는 빈 리스트 반환."""
        if not self.tasks_dispatched:
            self.tasks_dispatched = True
            return self.task_set
        return []

    # --- 호환성을 위한 나머지 함수들 ---
    def set_arm_gate(self, fn: Callable[[str, str], bool]):
        pass # 이 생성기에서는 사용하지 않음

    def complete_task(self, agv_id: int):
        pass # 필요 시 완료 카운트 로직 추가 가능

    def is_episode_done(self) -> bool:
        return False

    def get_progress(self) -> Dict:
        return {
            "spawned_total": len(self.task_set),
            "completed_total": 0, # 필요 시 구현
            "active_agvs": len(self.task_set),
            "max_agvs": len(self.task_set),
        }