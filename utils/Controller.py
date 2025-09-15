from utils.global_planning import AStar

class Planner:
    def __init__(self, map_data):
        """
        맵 데이터를 기반으로 경로 계획기(Planner)를 초기화합니다.
        """
        self.map = map_data

    def plan_path(self, start_pos, goal_pos):
        """
        A* 알고리즘을 사용하여 시작점에서 목표점까지의 경로를 계산하고 반환합니다.
        
        :param start_pos: 시작 좌표 (x, y)
        :param goal_pos: 목표 좌표 (x, y)
        :return: 좌표 리스트로 구성된 경로. 경로를 찾지 못하면 빈 리스트를 반환합니다.
        """
        if start_pos == goal_pos:
            return [start_pos]

        planner = AStar(self.map, start_pos, goal_pos)
        try:
            planner.compute_shortest_path()
            path = planner.extract_path()
            return path
        except Exception as e:
            print(f"Error during path planning from {start_pos} to {goal_pos}: {e}")
            # 경로 계획 실패 시, 제자리에 머무는 경로 반환
            return [start_pos]