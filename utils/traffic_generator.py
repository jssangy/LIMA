import random
import numpy as np
from typing import Dict, List, Optional, Tuple

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


DIRS = ("N", "E", "S", "W")

def discover_border_arms_3x3(intersections: Dict[str, object]) -> List[Tuple[str, str]]:
    """
    intersections: {iid(str) -> Intersection}이며 각 I가 center_x, center_y 속성을 가진다고 가정.
    3x3 그리드(가운데 제외)에서 맵 바깥을 향하는 12개 팔만 골라 반환.
    반환: [(iid, "N"/"E"/"S"/"W"), ...] 길이 12
    """
    # 1) 3개의 고유 x/y 추출 후 정렬
    xs = sorted({I.center_x for I in intersections.values()})
    ys = sorted({I.center_y for I in intersections.values()})
    assert len(xs) == 3 and len(ys) == 3, "교차로가 3x3이 아니거나 좌표가 비정상"

    xrank = {x: i for i, x in enumerate(xs)}        # 0:왼, 1:중앙, 2:오른
    yrank = {y: j for j, y in enumerate(ys)}        # 0:위, 1:중앙, 2:아래  (y가 위->아래로 증가한다고 가정)

    # 2) 각 교차로의 그리드 인덱스 계산
    grid = {}  # (i,j) -> iid
    for iid, I in intersections.items():
        i = xrank[I.center_x]
        j = yrank[I.center_y]
        grid[(i, j)] = iid

    # 3) 외곽 8개 교차로에서 바깥을 향하는 팔만 수집 (코너는 2개 팔)
    arms: List[Tuple[str, str]] = []
    for (i, j), iid in grid.items():
        if (i == 1 and j == 1):
            continue  # 중앙 제외

        # 위쪽 행이면 북(N) 팔이 외곽
        if j == 0:
            arms.append((iid, "N"))
        # 아래쪽 행이면 남(S) 팔이 외곽
        if j == 2:
            arms.append((iid, "S"))
        # 왼쪽 열이면 서(W) 팔이 외곽
        if i == 0:
            arms.append((iid, "W"))
        # 오른쪽 열이면 동(E) 팔이 외곽
        if i == 2:
            arms.append((iid, "E"))

    # sanity check: 정확히 12개여야 함
    assert len(arms) == 12, f"외곽 팔이 {len(arms)}개입니다(12가 아님). y축 방향이 반대라면 위/아래 매핑을 바꾸세요."
    return arms


class TrafficGenerator12:
    """
    12개 외곽 팔에서만 스폰하는 멀티-교차로 푸아송 생성기.
    - 각 arm별 arrivals ~ Poisson(lambda_arm[(iid,dir)] or lam)
    - 한 스텝에 k개가 오면 k개 전부 생성 (게이트/상한 없음)
    - goal은 12개 팔 중에서 '출발 팔을 제외한 11개'에서 균등 샘플
    """
    def __init__(
        self,
        arms12: List[Tuple[str, str]],
        lam: float = 0.01,
        lambda_per_arm: Optional[Dict[Tuple[str, str], float]] = None,
        seed: Optional[int] = None,
    ):
        self.rng = np.random.default_rng(seed)
        self.arms = [(str(iid), d) for (iid, d) in arms12]
        self.lam = float(lam)
        self.lambda_arm = {
            (iid, d): (lambda_per_arm[(iid, d)] if (lambda_per_arm and (iid, d) in lambda_per_arm) else self.lam)
            for (iid, d) in self.arms
        }
        self.agv_id_counter = 0
        self.spawned_total = 0
        self.completed_total = 0
        self.step_count = 0

        # 미리 후보목록 준비
        self._arm_idx = {a: k for k, a in enumerate(self.arms)}

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
        """
        반환 task:
          {
            "id": int,
            "intersection_id": str, "start_direction": "N|E|S|W",
            "goal_intersection_id": str, "goal_direction": "N|E|S|W"
          }
        """
        self.step_count += 1
        tasks: List[Dict] = []

        arms_rr = list(self.arms)
        random.shuffle(arms_rr)
        for (iid, d) in arms_rr:
            k = int(self.rng.poisson(self.lambda_arm[(iid, d)]))
            if k <= 0:
                continue
            for _ in range(k):
                gid, gd = self._sample_goal_excluding((iid, d))
                tasks.append({
                    "id": self.agv_id_counter,
                    "intersection_id": iid,
                    "start_direction": d,
                    "goal_intersection_id": gid,
                    "goal_direction": gd,
                })
                self.agv_id_counter += 1
                self.spawned_total += 1

        return tasks

    def complete_task(self, agv_id: int):
        self.completed_total += 1

    def is_episode_done(self) -> bool:
        return False

    def get_progress(self) -> Dict:
        return {
            "spawned_total": self.spawned_total,
            "completed_total": self.completed_total,
            "step": self.step_count,
            "arms": list(self.arms),
            "lambdas": {f"{iid}:{d}": self.lambda_arm[(iid, d)] for (iid, d) in self.arms},
        }

    # --- 내부 ---
    def _sample_goal_excluding(self, src_arm: Tuple[str, str]) -> Tuple[str, str]:
        # 12개 목록에서 src_arm만 제외하고 균등 샘플
        # (성능상 인덱스 운용)
        src_idx = self._arm_idx[src_arm]
        n = len(self.arms)  # 12
        # 0..n-2 중 하나 뽑고, src보다 크면 +1 해서 건너뛰기
        j = int(self.rng.integers(n - 1))
        if j >= src_idx:
            j += 1
        return self.arms[j]
