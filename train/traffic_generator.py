import random


class TrafficGenerator:
    """실험 5.1 전용 에피소드 기반 트래픽 생성기"""
    
    def __init__(self):
        self.directions = ['N', 'E', 'S', 'W']
        
        # 12개 경로 조합 생성 (동일 방향 제외)
        self.all_routes = []
        for start in self.directions:
            for goal in self.directions:
                if start != goal:
                    self.all_routes.append({
                        'start_direction': start,
                        'goal_direction': goal
                    })
        
        print(f"Total routes: {len(self.all_routes)}")
        for i, route in enumerate(self.all_routes):
            print(f"  Route {i+1}: {route['start_direction']} → {route['goal_direction']}")
        
        self.reset_episode()
    
    def start_new_episode(self):
        """새 에피소드 시작 - 12개 경로를 셔플"""
        # 12개 경로를 셔플
        self.current_episode_routes = self.all_routes.copy()
        random.shuffle(self.current_episode_routes)
        
        # 에피소드 상태 초기화
        self.current_task_index = 0
        self.completed_tasks = []
        
        print(f"\n=== New Episode Started ===")
        print(f"Episode route order:")
        for i, route in enumerate(self.current_episode_routes):
            print(f"  Task {i+1}: {route['start_direction']} → {route['goal_direction']}")
        print("=" * 30)
    
    def get_next_task(self):
        """다음 작업 정보 반환"""
        if self.current_task_index >= len(self.current_episode_routes):
            return None
        
        task_info = self.current_episode_routes[self.current_task_index].copy()
        task_info['task_id'] = self.current_task_index
        task_info['episode_task_number'] = self.current_task_index + 1
        
        return task_info
    
    def complete_current_task(self, current_time, success=True):
        """현재 작업 완료 처리"""
        if self.current_task_index >= len(self.current_episode_routes):
            return None
        
        task_info = self.current_episode_routes[self.current_task_index].copy()
        task_info.update({
            'task_id': self.current_task_index,
            'completion_time': current_time,
            'success': success,
            'duration': current_time  # 간단히 현재 시간을 duration으로 사용
        })
        
        self.completed_tasks.append(task_info)
        self.current_task_index += 1
        
        print(f"Task {self.current_task_index}/12 completed: {task_info['start_direction']} → {task_info['goal_direction']} "
              f"(Time: {task_info['completion_time']})")
        
        return task_info
    
    def has_remaining_tasks(self):
        """남은 작업이 있는지 확인"""
        return self.current_task_index < len(self.current_episode_routes)
    
    def get_progress(self):
        """에피소드 진행률 반환"""
        return {
            'completed_tasks': self.current_task_index,
            'total_tasks': len(self.current_episode_routes),
            'progress_ratio': self.current_task_index / len(self.current_episode_routes)
        }
    
    def get_episode_metrics(self):
        """에피소드 완료 시 성능 지표 반환"""
        if not self.completed_tasks:
            return {}
        
        # 성공한 작업들만 필터링
        successful_tasks = [task for task in self.completed_tasks if task['success']]
        
        metrics = {
            'total_tasks': len(self.completed_tasks),
            'successful_tasks': len(successful_tasks),
            'success_rate': len(successful_tasks) / len(self.completed_tasks) if self.completed_tasks else 0,
            'completed_routes': self.completed_tasks.copy()
        }
        
        if successful_tasks:
            completion_times = [task['completion_time'] for task in successful_tasks]
            metrics.update({
                'average_completion_time': sum(completion_times) / len(completion_times),
                'min_completion_time': min(completion_times),
                'max_completion_time': max(completion_times),
                'total_episode_time': max(completion_times)
            })
        
        return metrics
    
    def reset_episode(self):
        """에피소드 리셋"""
        self.current_episode_routes = []
        self.current_task_index = 0
        self.completed_tasks = []
    
    def get_current_task_info(self):
        """현재 작업 정보 반환"""
        if self.current_task_index >= len(self.current_episode_routes):
            return None
        return self.current_episode_routes[self.current_task_index]


if __name__ == "__main__":
    # 테스트 코드
    print("=== Episode-based Traffic Generator Test ===")
    
    generator = TrafficGenerator()
    
    # 에피소드 시뮬레이션
    print("\n=== Episode Simulation ===")
    generator.start_new_episode()
    
    current_time = 0
    while generator.has_remaining_tasks():
        # 다음 작업 가져오기
        task = generator.get_next_task()
        print(f"\nTime {current_time}: Starting task {task['episode_task_number']}: "
              f"{task['start_direction']} → {task['goal_direction']}")
        
        # 작업 완료 시뮬레이션 (8-15초 소요)
        duration = random.randint(8, 15)
        current_time += duration
        
        # 작업 완료
        completed = generator.complete_current_task(current_time, success=True)
        
        # 진행률 출력
        progress = generator.get_progress()
        print(f"Progress: {progress['completed_tasks']}/{progress['total_tasks']} "
              f"({progress['progress_ratio']*100:.1f}%)")
    
    # 에피소드 완료 지표
    print("\n=== Episode Completed ===")
    metrics = generator.get_episode_metrics()
    print(f"Episode metrics: {metrics}")
    
    print(f"\nSuccess Rate: {metrics['success_rate']*100:.1f}%")
    print(f"Average Completion Time: {metrics['average_completion_time']:.1f}s")
    print(f"Total Episode Time: {metrics['total_episode_time']}s")
    
    # 두 번째 에피소드 테스트 (다른 순서)
    print("\n" + "="*50)
    print("=== Second Episode (Different Order) ===")
    generator.start_new_episode()
    
    print("\nFirst 3 tasks of second episode:")
    for i in range(3):
        task = generator.get_next_task()
        if task:
            print(f"  Task {task['episode_task_number']}: {task['start_direction']} → {task['goal_direction']}")
            generator.complete_current_task(i*10, success=True)