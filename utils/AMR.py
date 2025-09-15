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

        # 자율 주행 및 경로 복귀를 위한 상태
        self.path = []
        self.path_cursor = 0
        self.off_path = False

        # AMR가 스스로 계산하는 다음 이동 제어 신호
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

    def update_next_buffer(self):
        """
        자신의 이상적인 계획을 next_buffer에만 기록합니다.
        [수정] 경로 이탈 시, 남은 경로 중 가장 가까운 직선(수평/수직) 도달 가능 지점으로 복귀합니다.
        """
        # 경로 이탈 상태일 경우
        if self.off_path:
            closest_reachable_point = None
            min_dist = float('inf')
            current_pos = self.pos

            # 남은 경로(path[path_cursor:])를 순회하며 가장 가까운 직선 도달 가능 지점을 찾음
            for path_point in self.path[self.path_cursor:]:
                # 수평 또는 수직으로 일치하는지 확인 (직선 도달 가능 조건)
                if path_point[0] == current_pos[0] or path_point[1] == current_pos[1]:
                    dist = abs(path_point[0] - current_pos[0]) + abs(path_point[1] - current_pos[1])
                    if dist < min_dist:
                        min_dist = dist
                        closest_reachable_point = path_point
            
            # 가장 가까운 직선 도달 가능 지점을 찾았다면, 그 방향으로 이동
            if closest_reachable_point:
                dx = closest_reachable_point[0] - current_pos[0]
                dy = closest_reachable_point[1] - current_pos[1]
                
                # np.sign을 사용하여 한 칸 이동 방향 결정
                self.next_buffer = (np.sign(dx), np.sign(dy))
            return

        # 정상적으로 경로를 따라가는 경우
        if self.path_cursor < len(self.path) - 1:
            current_target = self.path[self.path_cursor + 1]
            dx = np.sign(current_target[0] - self.pos[0])
            dy = np.sign(current_target[1] - self.pos[1])
            self.next_buffer = (dx, dy)
        else:
            # 경로의 끝에 도달했으면 정지
            self.next_buffer = (0, 0)

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
        self.priority = 0 # 우선순위는 매 스텝 초기화

        try:
            # 현재 위치가 경로상의 몇 번째 인덱스에 있는지 찾음
            current_idx_on_path = self.path.index(self.pos)
            
            # 성공적으로 찾았다면, path_cursor를 해당 인덱스로 업데이트하고 off_path를 해제
            self.path_cursor = current_idx_on_path
            self.off_path = False

        except ValueError:
            # 현재 위치가 경로상에 존재하지 않으면, 경로 이탈 상태로 설정
            self.off_path = True
