import random

class TrafficGenerator:    
    def __init__(self, spawn_policy='completion', spawn_interval=50, total_tasks=12):
        if spawn_policy not in ['time', 'completion']:
            raise ValueError("spawn_policy must be either 'time' or 'completion'")
        
        self.spawn_policy = spawn_policy
        self.spawn_interval = spawn_interval
        self.total_tasks_in_episode = total_tasks
        
        self.directions = ['N', 'E', 'S', 'W']
        self.all_routes = []
        for start in self.directions:
            for goal in self.directions:
                if start != goal:
                    self.all_routes.append({
                        'start_direction': start,
                        'goal_direction': goal
                    })
        
        self.agv_id_counter = 0
        self.spawn_trigger = False # 'completion' 정책을 위한 트리거

    def start_new_episode(self):
        """새 에피소드 시작"""
        self.tasks_to_spawn = random.sample(self.all_routes, self.total_tasks_in_episode)
        
        self.active_tasks = {}
        self.completed_task_count = 0
        self.next_spawn_time = 0
        self.agv_id_counter = 0
        
        # [추가] 'completion' 정책일 경우, 첫 AGV를 즉시 생성하도록 트리거 설정
        if self.spawn_policy == 'completion':
            self.spawn_trigger = True
        else:
            self.spawn_trigger = False

        print(f"\n=== New Episode Started (Policy: {self.spawn_policy}) with {self.total_tasks_in_episode} tasks ===")

    def should_spawn_next(self, current_time):
        """새 AGV를 생성할 시간인지, 그리고 생성할 작업이 남았는지 확인"""
        has_tasks_left = bool(self.tasks_to_spawn)
        if not has_tasks_left:
            return False

        # [수정] 정책에 따라 스폰 조건 분기
        if self.spawn_policy == 'time':
            return current_time >= self.next_spawn_time
        elif self.spawn_policy == 'completion':
            return self.spawn_trigger
        
        return False

    def get_next_task(self):
        """다음 생성할 AGV의 정보를 반환"""
        if not self.tasks_to_spawn:
            return None
        
        task_info = self.tasks_to_spawn.pop(0)
        
        self.agv_id_counter += 1
        agv_id = f"agv_{self.agv_id_counter}"
        task_info['id'] = agv_id
        
        # [수정] 정책에 따라 다음 스폰 시간 또는 트리거를 업데이트
        if self.spawn_policy == 'time':
            self.next_spawn_time += self.spawn_interval
        elif self.spawn_policy == 'completion':
            self.spawn_trigger = False # 스폰 후에는 트리거를 비활성화
        
        self.active_tasks[agv_id] = task_info
        return task_info

    def complete_task(self, agv_id, current_time, success=True):
        """특정 AGV의 작업 완료 처리"""
        if agv_id in self.active_tasks:
            task_info = self.active_tasks.pop(agv_id)
            self.completed_task_count += 1
            
            # [추가] 'completion' 정책일 경우, 다음 스폰을 위한 트리거 설정
            if self.spawn_policy == 'completion':
                self.spawn_trigger = True

            print(f"Task completed for {agv_id} ({task_info['start_direction']}->{task_info['goal_direction']}). "
                  f"Progress: {self.completed_task_count}/{self.total_tasks_in_episode}")
        else:
            print(f"[Warning] Trying to complete a task for an unknown AGV ID: {agv_id}")

    def is_episode_done(self):
        """모든 작업이 생성되고, 모든 활성 AGV가 작업을 완료했는지 확인"""
        all_tasks_spawned = not self.tasks_to_spawn
        no_active_agvs = not self.active_tasks
        return all_tasks_spawned and no_active_agvs

    def get_progress(self):
        """에피소드 진행률 반환"""
        return {
            'completed_tasks': self.completed_task_count,
            'total_tasks': self.total_tasks_in_episode,
            'active_agvs': len(self.active_tasks)
        }