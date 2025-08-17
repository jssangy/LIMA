import random

class TrafficGenerator:
    """
    두 대의 AMR이 한 쌍으로 움직이는 시나리오를 생성합니다.
    한 쌍의 AMR이 모두 목적지에 도달하면, 다음 교차 경로를 가진 한 쌍이 생성됩니다.
    """
    def __init__(self):
        # 6가지 교차(crossing) 작업 쌍을 정의
        self.task_pairs = [
            [{'start_direction': 'N', 'goal_direction': 'S'}, {'start_direction': 'S', 'goal_direction': 'N'}],
            [{'start_direction': 'E', 'goal_direction': 'W'}, {'start_direction': 'W', 'goal_direction': 'E'}],
            [{'start_direction': 'N', 'goal_direction': 'E'}, {'start_direction': 'E', 'goal_direction': 'N'}],
            [{'start_direction': 'N', 'goal_direction': 'W'}, {'start_direction': 'W', 'goal_direction': 'N'}],
            [{'start_direction': 'E', 'goal_direction': 'S'}, {'start_direction': 'S', 'goal_direction': 'E'}],
            [{'start_direction': 'W', 'goal_direction': 'S'}, {'start_direction': 'S', 'goal_direction': 'W'}],
        ]
        self.total_tasks_in_episode = len(self.task_pairs)
        self.agv_id_counter = 0
        self.spawn_trigger = False

    def start_new_episode(self):
        """새 에피소드 시작"""
        # 정의된 작업 쌍의 순서를 무작위로 섞음
        self.task_pairs_to_spawn = random.sample(self.task_pairs, len(self.task_pairs))
        
        self.active_pair_agv_ids = set()
        self.completed_pair_count = 0
        self.agv_id_counter = 0
        
        # 첫 번째 쌍을 즉시 생성하도록 트리거 설정
        self.spawn_trigger = True

        print(f"\n=== New Episode Started with {self.total_tasks_in_episode} task pairs ===")

    def should_spawn_next(self):
        """새로운 AMR 쌍을 생성할지 확인"""
        # 생성할 작업 쌍이 남아있고, 생성 트리거가 켜져있을 때
        return bool(self.task_pairs_to_spawn) and self.spawn_trigger

    def get_next_task_pair(self):
        """다음 생성할 AMR 쌍(2개)의 정보를 반환"""
        if not self.task_pairs_to_spawn:
            return None
        
        # 생성할 작업 쌍 목록에서 하나를 꺼냄
        task_pair_info = self.task_pairs_to_spawn.pop(0)
        
        # 각 작업에 고유 ID 할당
        task1_info = task_pair_info[0]
        task1_info['id'] = self.agv_id_counter
        self.agv_id_counter += 1
        
        task2_info = task_pair_info[1]
        task2_info['id'] = self.agv_id_counter
        self.agv_id_counter += 1

        # 현재 활성화된 쌍의 ID들을 기록
        self.active_pair_agv_ids = {task1_info['id'], task2_info['id']}
        
        # 생성 후 트리거 비활성화
        self.spawn_trigger = False
        
        return [task1_info, task2_info]

    def complete_task(self, agv_id):
        """특정 AGV의 작업 완료 처리"""
        if agv_id in self.active_pair_agv_ids:
            # 완료된 AGV를 활성 쌍에서 제거
            self.active_pair_agv_ids.remove(agv_id)
            
            # 만약 활성 쌍의 모든 AGV가 작업을 완료했다면
            if not self.active_pair_agv_ids:
                self.completed_pair_count += 1
                # 다음 쌍 생성을 위한 트리거 설정
                self.spawn_trigger = True
                print(f"--- Task Pair {self.completed_pair_count}/{self.total_tasks_in_episode} completed! ---")
        else:
            print(f"[Warning] Trying to complete a task for an unknown or already completed AGV ID: {agv_id}")

    def is_episode_done(self):
        """모든 작업 쌍이 완료되었는지 확인"""
        return self.completed_pair_count >= self.total_tasks_in_episode

    def get_progress(self):
        """에피소드 진행률 반환"""
        return {
            'completed_pairs': self.completed_pair_count,
            'total_pairs': self.total_tasks_in_episode,
            'active_agvs': len(self.active_pair_agv_ids)
        }