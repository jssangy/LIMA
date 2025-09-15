from utils.global_planning import AStar, PIBT

class AStarPlanner:
    def __init__(self, map_data):
        """
        맵 데이터를 기반으로 경로 계획기(Planner)를 초기화합니다.
        """
        self.map = map_data

    def plan_path(self, amr_list):
        """
        [수정] AMR 객체들의 딕셔너리를 받아, 각각의 경로를 계산하고 주입합니다.
        
        :param amr_list: AMR 객체들을 담고 있는 딕셔너리 {amr_id: AMR_Object}
        """
        for amr in amr_list.values():
            if amr.path:
                continue

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
                    path = [start_pos] # 실패 시 제자리 경로
            
            # 계산된 경로를 AMR 객체에 직접 주입
            amr.set_path(path)

class PIBTPlanner:
    def __init__(self, map_data):
        """
        맵 데이터를 기반으로 PIBT 플래너를 초기화합니다.
        """
        self.map = map_data

    def plan_path(self, amr_list):
        pass

class CBSPlanner:
    def __init__(self, map_data):
        """
        맵 데이터를 기반으로 CBS 플래너를 초기화합니다.
        """
        self.map = map_data

    def plan_path(self, amr_list):
        pass