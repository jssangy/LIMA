import random

class TrafficGenerator:
    """
    두 대의 AMR이 한 쌍으로 움직이는 시나리오를 '무한 스트리밍'으로 생성합니다.
    6개 작업 쌍을 모두 완료하면 즉시 새로 섞어서 다음 6개 작업 쌍을 이어서 생성합니다.
    에피소드 종료는 시간 제한 등 ENV 쪽에서 관리하세요.
    """
    def __init__(self):
        # 기본 6가지 교차(crossing) 작업 쌍 정의
        self.base_task_pairs = [
            [{'start_direction': 'N', 'goal_direction': 'S'}, {'start_direction': 'S', 'goal_direction': 'N'}],
            [{'start_direction': 'E', 'goal_direction': 'W'}, {'start_direction': 'W', 'goal_direction': 'E'}],
            [{'start_direction': 'N', 'goal_direction': 'E'}, {'start_direction': 'E', 'goal_direction': 'N'}],
            [{'start_direction': 'N', 'goal_direction': 'W'}, {'start_direction': 'W', 'goal_direction': 'N'}],
            [{'start_direction': 'E', 'goal_direction': 'S'}, {'start_direction': 'S', 'goal_direction': 'E'}],
            [{'start_direction': 'W', 'goal_direction': 'S'}, {'start_direction': 'S', 'goal_direction': 'W'}],
        ]
        self.total_tasks_in_episode = len(self.base_task_pairs)  # 호환성 유지(색상맵 등에서 사용)
        self.agv_id_counter = 0

        # 런타임 상태
        self.task_pairs_to_spawn = []      # 현재 사이클에서 남은 작업쌍
        self.active_pair_agv_ids = set()   # 현재 활성 쌍의 AGV id 집합(2개)
        self.completed_pair_count = 0      # 현재 사이클에서 완료한 쌍 수
        self.completed_pair_count_total = 0# 누적 완료 쌍 수(모든 사이클)
        self.cycle = 0                     # 완료한 사이클 수
        self.spawn_trigger = False         # 다음 쌍 생성 트리거

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
            'completed_pairs_total': self.completed_pair_count_total,
            'active_agvs': len(self.active_pair_agv_ids),
            'cycle': self.cycle,
        }
