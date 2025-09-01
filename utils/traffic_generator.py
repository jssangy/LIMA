import random
import numpy as np
from typing import Dict, List, Optional

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

