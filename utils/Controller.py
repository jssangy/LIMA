from utils.global_planning import AStar, PIBT

class Planner:
    def __init__(self, map_data):
        """
        맵 데이터를 기반으로 경로 계획기(Planner)를 초기화합니다.
        """
        self.map = map_data

    def plan_path(self, amr_list):
        """
        [수정] AMR 객체들의 딕셔너리를 받아, 각각의 경로를 계산하고 주입합니다.
        향후 이 함수 내부가 CBS와 같은 전역 플래너로 대체될 수 있습니다.
        
        :param amr_list: AMR 객체들을 담고 있는 딕셔너리 {amr_id: AMR_Object}
        """
        # TODO: 향후 이 부분을 CBS 플래너 호출로 대체
        for amr in amr_list.values():
            # 이미 경로가 있는 AMR은 건너뜁니다 (예: 기존에 있던 AMR).
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