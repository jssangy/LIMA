from utils.global_planning import AStar, PIBT, BFS, CBS

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
        """
        [역할 정의]
        CBS는 전역 플래너이므로, 새로운 AMR이 추가되면 모든 AMR의 경로를
        다시 계산해야 충돌 없음을 보장할 수 있습니다.
        따라서 이 함수는 replan_all 함수를 호출합니다.
        """
        # 경로가 없는 AMR이 있는지 확인
        for amr in amr_list.values():
            # 경로가 이미 있는 AMR은 건너뜁니다.
            if amr.path:
                continue
            else:
                self.replan_all(amr_list)
                break

        # needs_replan = any(not amr.path for amr in amr_list.values())
        
        # if needs_replan:
        #     print("CBS Planner: 새로운 AMR이 감지되어 모든 에이전트의 경로를 다시 계산합니다.")
        #     self.replan_all(amr_list)

    def replan_all(self, amr_list):
        """
        [역할 정의]
        현재 있는 모든 AMR에 대해 CBS 알고리즘을 실행하여
        충돌 없는 전체 경로를 계산하고 각 AMR에 주입합니다.
        """
        if not amr_list:
            print("CBS Planner: 경로를 계산할 AMR이 없습니다.")
            return

        agents_to_plan = {}
        for amr in amr_list.values():
            # 각 AMR의 상태를 초기화합니다.
            # amr.reset()
            start_pos = (int(amr.pos[0]), int(amr.pos[1]))
            goal_pos = (int(amr.goal[0]), int(amr.goal[1]))
            # solver에 필요한 데이터를 준비합니다.
            agents_to_plan[amr.id] = {'start': start_pos, 'goal': goal_pos}

        # 2. CBS solver 초기화 및 실행
        cbs_solver = CBS(self.map, agents_to_plan)
        timeout_seconds = 10.0  # 필요시 이 시간(초)을 조절하세요.
        print('Trying to calculate within', timeout_seconds, 'seconds...')
        solution = cbs_solver.solve(time_limit=timeout_seconds)

        # 3. 반환된 solution을 기반으로 각 AMR에 경로 설정
        if solution:
            # 모든 AMR에 대해 계산된 경로를 주입
            for amr_id, path in solution.items():
                if amr_id in amr_list:
                    amr_list[amr_id].set_path(path)
                    print('amr 번호', amr_id, '/ 경로:', path)
            
            # solution에 포함되지 않은 AMR이 있을 경우 (드문 경우), 현재 위치에 머무르도록 설정
            for amr in amr_list.values():
                if not amr.path:
                    amr.set_path([amr.pos])
        else:
            print("CBS Planner: 해결책을 찾지 못했습니다.")
            # 해결책을 못 찾은 경우, 모든 AMR은 오류 방지를 위해 현재 위치에 머무르는 경로를 가짐
            for amr in amr_list.values():
                amr.set_path([amr.pos])


    def calculate_and_set_path(self, amr):
        """
        [역할 정의]
        이 메서드는 단일 에이전트에 대한 경로 계산을 의미하지만, CBS는
        다중 에이전트를 동시에 고려하는 전역 플래너이므로 이 메서드는 사용하지 않습니다.
        """
        pass

class BFSPlanner:
    def __init__(self, map_data):
        """
        맵 데이터를 기반으로 BFS 기반 거리장 플래너를 초기화합니다.
        """
        self.planner = BFS(map_data)

    def plan_for_new_amrs(self, amr_list):
        """
        경로가 없는 새로운 AMR에 대해서만 경로를 계산합니다.
        BFS 플래너는 매우 빠르므로, 모든 경로를 다시 계산해도 성능 저하가 거의 없습니다.
        """
        for amr in amr_list.values():
            if not amr.path:
                self.calculate_and_set_path(amr)

    def replan_all(self, amr_list):
        """
        모든 AMR의 경로를 강제로 다시 계산합니다.
        """
        for amr in amr_list.values():
            self.calculate_and_set_path(amr)

    def calculate_and_set_path(self, amr):
        """
        BFS(거리장) 플래너를 사용하여 경로를 계산하고 주입합니다.
        """
        path = self.planner.plan_path(amr.pos, amr.goal)
        amr.set_path(path)