import numpy as np

# AMR Object
class AMR():
    def __init__(self, pos, goal, id, color):
        # 기본 정보
        self.id = id
        self.color = color
        self.goal = goal
        self.pos = pos
        self.prev_pos = pos
        self.steps = 0
        self.priority = 0       # 숫자가 클수록 우선순위가 높음 (AMR 충돌 해결용)
        self.current_intersection_id = set() # [추가] 현재 속한 교차로 ID


        # 자율 주행 및 경로 복귀를 위한 상태
        self.path = []
        self.path_cursor = 0
        self.off_path = False

        # AMR가 스스로 계산하는 다음 이동 제어 신호
        self.next_buffer = (0, 0)
        self.control_buffer = (0, 0)

    def reset(self):
        """
        AMR의 상태를 초기화합니다.
        """
        self.steps = 0
        self.priority = 0
        self.current_intersection_id.clear()

        self.path = []
        self.path_cursor = 0
        self.off_path = False

        self.next_buffer = (0, 0)
        self.control_buffer = (0, 0)

    def set_path(self, new_path: list):
        """
        규칙 1: 플래너로부터 새로운 전체 경로를 주입받고 상태를 초기화합니다.
        """
        if not new_path:
            self.path = [self.pos] # 경로가 없으면 제자리에 머무는 경로로 설정
        else:
            self.path = new_path

        self.path_cursor = 0
        self.off_path = False

        self.update_next_buffer()

    def update_next_buffer(self, env_map=None, intersection=None):
        """
        [전면 수정] 경로 이탈 시, 교차로 내부/외부 상황에 따라 복귀 전략을 다르게 설정합니다.
        """
        # 경로 이탈 상태이고, map 정보가 주어졌을 경우
        if self.off_path and env_map is not None:
            self.next_buffer = self._recover_inside_intersection(intersection)

        # 정상적으로 경로를 따라가는 경우
        if self.path_cursor < len(self.path) - 1:
            current_target = self.path[self.path_cursor + 1]
            dx = np.sign(current_target[0] - self.pos[0])
            dy = np.sign(current_target[1] - self.pos[1])
            self.next_buffer = (dx, dy)
        else:
            # 경로의 끝에 도달했으면 정지
            self.next_buffer = (0, 0)

    def _recover_inside_intersection(self, intersection):
        """
        [신규] 교차로 내부에서 경로를 이탈했을 때의 복귀 로직입니다.
        """
        # 1. 나의 예측 출구 팔(exit arm)을 확인합니다.
        predicted_exit_arm = intersection._get_exit_arm_from_goal(self.goal)
        
        # 2. 나의 현재 위치가 어느 팔(arm)에 있는지 확인합니다.
        current_arm = intersection._get_arm_from_pos(self.pos)

        # 3. 현재 팔과 예측 출구 팔을 비교하여 행동을 결정합니다.
        # 3-a: 잘못된 팔에 있다면, 무조건 교차로 중앙으로 이동
        if current_arm and current_arm != predicted_exit_arm:
            center_pos = (intersection.center_x, intersection.center_y)
            dx = np.sign(center_pos[0] - self.pos[0])
            dy = np.sign(center_pos[1] - self.pos[1])
            return (dx, dy)
        # 3-b: 올바른 팔에 있거나, 중앙에 있거나, 위치를 특정할 수 없다면
        #     최종 목표를 향해 이동 (기본 전략)
        else:
            # _find_best_step_to_goal 로직을 여기서 직접 수행
            best_move = (0, 0)
            min_dist_to_goal = abs(self.pos[0] - self.goal[0]) + abs(self.pos[1] - self.goal[1])
            for move in [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]:
                next_pos = (self.pos[0] + move[0], self.pos[1] + move[1])
                dist_to_goal = abs(next_pos[0] - self.goal[0]) + abs(next_pos[1] - self.goal[1])
                if dist_to_goal < min_dist_to_goal:
                    min_dist_to_goal = dist_to_goal
                    best_move = move
            return best_move

    def move(self, final_control_signal):
        """
        [수정] 이동 후, 경로상의 현재 위치를 찾아 path_cursor를 동기화합니다.
        """
        if final_control_signal == (0, 0):
            self.prev_pos = self.pos
            self.priority = 0
            return

        self.prev_pos = self.pos
        self.pos = (self.pos[0] + final_control_signal[0], self.pos[1] + final_control_signal[1])
        self.steps += 1
        self.priority = 0                           # 우선순위는 매 스텝 초기화
        self.current_intersection_id.clear()        # [추가] 현재 속한 교차로 ID 초기화

        try:
            # 현재 위치가 경로상의 몇 번째 인덱스에 있는지 찾음
            current_idx_on_path = self.path.index(self.pos)
            
            # 성공적으로 찾았다면, path_cursor를 해당 인덱스로 업데이트하고 off_path를 해제
            self.path_cursor = current_idx_on_path
            self.off_path = False

        except ValueError:
            # 현재 위치가 경로상에 존재하지 않으면, 경로 이탈 상태로 설정
            self.off_path = True
