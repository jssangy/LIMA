import random
import numpy as np
from typing import Dict, List, Optional

class TrafficGenerator1:
    """
    두 대의 AMR이 한 쌍으로 움직이는 시나리오를 '무한 스트리밍'으로 생성합니다.
    6개 작업 쌍을 모두 완료하면 즉시 새로 섞어서 다음 6개 작업 쌍을 이어서 생성합니다.
    에피소드 종료는 시간 제한 등 ENV 쪽에서 관리하세요.
    """
    def __init__(self):
        # 기본 6가지 교차(crossing) 작업 쌍 정의
        self.base_task_pairs = [
            [{'start_direction': 'S', 'goal_direction': 'N'}, {'start_direction': 'N', 'goal_direction': 'S'}],
            [{'start_direction': 'E', 'goal_direction': 'W'}, {'start_direction': 'W', 'goal_direction': 'E'}],
            [{'start_direction': 'E', 'goal_direction': 'N'}, {'start_direction': 'N', 'goal_direction': 'E'}],
            [{'start_direction': 'N', 'goal_direction': 'W'}, {'start_direction': 'W', 'goal_direction': 'N'}],
            [{'start_direction': 'S', 'goal_direction': 'E'}, {'start_direction': 'E', 'goal_direction': 'S'}],
            [{'start_direction': 'W', 'goal_direction': 'S'}, {'start_direction': 'S', 'goal_direction': 'W'}],
        ]
        self.total_tasks_in_episode = len(self.base_task_pairs)  # 호환성 유지(색상맵 등에서 사용)
        self.agv_id_counter = 0

        # 런타임 상태
        self.task_pairs_to_spawn = []           # 현재 사이클에서 남은 작업쌍
        self.active_pair_agv_ids = set()        # 현재 활성 쌍의 AGV id 집합(2개)
        self.completed_pair_count = 0           # 현재 사이클에서 완료한 쌍 수
        self.completed_pair_count_total = 0     # 누적 완료 쌍 수(모든 사이클)
        self.cycle = 0                          # 완료한 사이클 수
        self.spawn_trigger = False              # 다음 쌍 생성 트리거

    def _start_new_cycle(self, reset_counters: bool):
        """새 사이클 시작(6개를 섞어서 큐에 적재)"""
        self.task_pairs_to_spawn = random.sample(self.base_task_pairs, len(self.base_task_pairs))
        self.active_pair_agv_ids.clear()
        self.spawn_trigger = True  # 사이클 시작 즉시 첫 쌍을 생성할 수 있도록
        if reset_counters:
            self.completed_pair_count = 0
            self.completed_pair_count_total = 0
            self.cycle = 0
            self.agv_id_counter = 0  # 에피소드 리셋 시에만 ID 리셋
        else:
            self.completed_pair_count = 0  # 사이클 로컬 카운터만 리셋

    def start_new_episode(self):
        """ENV.reset()에서 호출: 스트리밍 에피소드 시작"""
        self._start_new_cycle(reset_counters=True)

    def should_spawn_next(self) -> bool:
        """새로운 AMR 쌍을 생성할지 확인"""
        # 남은 쌍이 있고 트리거가 켜져 있으면 True
        return bool(self.task_pairs_to_spawn) and self.spawn_trigger

    def get_next_task_pair(self):
        """다음 생성할 AMR 쌍(2개)의 정보를 반환(없으면 None)"""
        if not self.task_pairs_to_spawn:
            return None
        # 큐에서 하나 꺼냄
        task_pair_info = self.task_pairs_to_spawn.pop(0)

        # 고유 ID 부여(에피소드 전체에서 유니크)
        task1 = dict(task_pair_info[0])
        task1['id'] = self.agv_id_counter; self.agv_id_counter += 1

        task2 = dict(task_pair_info[1])
        task2['id'] = self.agv_id_counter; self.agv_id_counter += 1

        # 현재 활성 쌍 ID 기록 및 트리거 off
        self.active_pair_agv_ids = {task1['id'], task2['id']}
        self.spawn_trigger = False

        return [task1, task2]

    def complete_task(self, agv_id: int):
        """특정 AGV의 작업 완료 처리"""
        if agv_id in self.active_pair_agv_ids:
            self.active_pair_agv_ids.remove(agv_id)
            # 쌍의 두 대가 모두 완료되면 다음 쌍 생성 트리거 on
            if not self.active_pair_agv_ids:
                self.completed_pair_count += 1
                self.completed_pair_count_total += 1
                self.spawn_trigger = True

                # 사이클 완료 시: 자동으로 다음 사이클 시작
                if self.completed_pair_count >= self.total_tasks_in_episode:
                    self.cycle += 1
                    self._start_new_cycle(reset_counters=False)
        else:
            print(f"[Warning] Unknown or already completed AGV ID: {agv_id}")

    def is_episode_done(self) -> bool:
        """스트리밍 모드: 에피소드 종료 없음(시간제한 등은 ENV에서 관리)"""
        return False

    def get_progress(self) -> dict:
        """진행 상황 리포트(로깅용)"""
        return {
            'completed_pairs_in_cycle': self.completed_pair_count,
            'total_pairs_per_cycle': self.total_tasks_in_episode,
            'completed_total': self.completed_pair_count_total,
            'active_agvs': len(self.active_pair_agv_ids),
            'cycle': self.cycle,
        }


DIRS = ("N", "E", "S", "W")

class TrafficGenerator2:
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
        lam: float = 0.05,                           # 기본 λ (per-step, per-dir)
        lambda_per_dir: Optional[Dict[str, float]] = None,
        max_spawn_per_step: int = 2,                 # 한 스텝 최대 스폰 수 (K)
        turn_probs: Optional[Dict[str, Dict[str, float]]] = None,
        seed: Optional[int] = None,
    ):
        self.rng = np.random.default_rng(seed)
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
        각 원소는 {'id', 'start_direction', 'goal_direction'}.
        """
        self.step_count += 1
        tasks: List[Dict] = []

        # 1) 새 유입 샘플(방향별 Poisson)
        arrivals = {d: int(self.rng.poisson(self.lambda_d[d])) for d in self.active_dirs}
        # 2) backlog에 누적
        for d in self.active_dirs:
            self.backlog[d] += arrivals[d]

        # 3) 라운드로빈으로 스폰(최대 K개), 용량 게이트가 있으면 통과해야 스폰
        spawned = 0
        # 시작 오프셋을 섞어서 방향 편향 방지
        dirs_rr = list(self.active_dirs)
        random.shuffle(dirs_rr)

        while spawned < self.max_spawn_per_step:
            progressed = False
            for d in dirs_rr:
                if spawned >= self.max_spawn_per_step:
                    break
                if self.backlog[d] <= 0:
                    continue
                if self._capacity_gate is not None and not self._capacity_gate(d):
                    continue  # 슬롯 없으면 다음 방향

                # 스폰 확정
                g = self._sample_goal(d)
                task = {"id": self.agv_id_counter,
                        "start_direction": d,
                        "goal_direction": g}
                tasks.append(task)
                self.agv_id_counter += 1
                self.backlog[d] -= 1
                spawned += 1
                self.spawned_total += 1
                progressed = True
            if not progressed:
                # 모든 방향이 backlog>0인데 용량게이트 통과 못한 경우 → 다음 스텝으로 미룸
                break

        return tasks  # []일 수도 있음

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