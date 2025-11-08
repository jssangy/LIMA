import numpy as np

# AMR Object
class AMR():
    def __init__(self, id, pos, goal, color):
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

        self.next_buffer = (self.path[1][0] - self.path[0][0], self.path[1][1] - self.path[0][1])
        self.control_buffer = (self.path[1][0] - self.path[0][0], self.path[1][1] - self.path[0][1])

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
            self.next_buffer = (self.path[1][0] - self.path[0][0], self.path[1][1] - self.path[0][1])

        except ValueError:
            # 현재 위치가 경로상에 존재하지 않으면, 경로 이탈 상태로 설정
            self.off_path = True