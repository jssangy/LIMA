from utils.global_planning import AStar, PIBT

class AStarPlanner:
    def __init__(self, map_data):
        """
        맵 데이터를 기반으로 경로 계획기(Planner)를 초기화합니다.
        """
        self.map = map_data

    def plan_for_new_amrs(self, amr_list):
        """
        [이름 변경 및 역할 명확화]
        경로가 없는 새로운 AMR에 대해서만 경로를 계산하고 주입합니다.
        """
        for amr in amr_list.values():
            # 경로가 이미 있는 AMR은 건너뜁니다.
            if amr.path:
                continue

            self.calculate_and_set_path(amr)

    def replan_all(self, amr_list):
        """
        [신규] 모든 AMR의 경로를 강제로 다시 계산하고 주입합니다.
        """
        for amr in amr_list.values():
            # 경로 존재 여부와 상관없이 무조건 다시 계산합니다.
            amr.reset()
            self.calculate_and_set_path(amr)

    def calculate_and_set_path(self, amr):
        """
        [신규] A* 경로 계산 및 주입 로직을 별도 함수로 분리 (코드 중복 방지)
        """
        start_pos = amr.pos
        goal_pos = amr.goal

        if start_pos == goal_pos:
            path = [start_pos]
        else:
            planner = AStar(self.map, start_pos, goal_pos)
            try:
                planner.compute_shortest_path()
                path = planner.extract_path()
            except Exception as e:
                print(f"Error during path planning for AMR {amr.id} from {start_pos} to {goal_pos}: {e}")
                path = [start_pos]
        
        amr.set_path(path)

class PIBTPlanner:
    def __init__(self, map_data):
        """
        맵 데이터를 기반으로 PIBT 플래너를 초기화합니다.
        """
        self.map = map_data

    def plan_for_new_amrs(self, amr_list):
        pass

    def replan_all(self, amr_list):
        pass

    def caculate_and_set_path(self, amr):
        pass

class CBSPlanner:
    def __init__(self, map_data):
        """
        맵 데이터를 기반으로 CBS 플래너를 초기화합니다.
        """
        self.map = map_data

    def plan_for_new_amrs(self, amr_list):
        pass

    def replan_all(self, amr_list):
        pass

    def caculate_and_set_path(self, amr):
        pass