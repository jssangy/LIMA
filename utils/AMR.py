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

        # AGV가 스스로 계산하는 다음 이동 제어 신호
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

        self.update_control_buffer()
    
    def update_control_buffer(self):
        """
        [새로운 함수] 현재 상태에 따라 다음 스텝의 제어 신호를 스스로 결정합니다.
        """
        # 경로 이탈 상태일 경우
        if self.off_path:
            # 규칙 5: 원래 경로(path[path_cursor])로 복귀하기 위한 방향 벡터 계산
            target_pos = self.path[self.path_cursor]
            dx = np.sign(target_pos[0] - self.pos[0])
            dy = np.sign(target_pos[1] - self.pos[1])
            self.control_buffer = (dx, dy)
            return

        # 정상적으로 경로를 따라가는 경우
        if self.path_cursor < len(self.path) - 1:
            # 규칙 2: 경로의 다음 지점으로 이동하기 위한 방향 벡터 계산
            current_target = self.path[self.path_cursor + 1]
            dx = np.sign(current_target[0] - self.pos[0])
            dy = np.sign(current_target[1] - self.pos[1])
            self.control_buffer = (dx, dy)
        else:
            # 경로의 끝에 도달했으면 정지
            self.control_buffer = (0, 0)

    def move(self, final_control_signal):
        """
        [수정] 최종 결정된 제어 신호로 이동하고, 자신의 상태를 업데이트합니다.
        """
        self.prev_pos = self.pos
        self.pos = (self.pos[0] + final_control_signal[0], self.pos[1] + final_control_signal[1])
        self.steps += 1
        self.priority = 0 # 우선순위는 매 스텝 초기화

        # --- 상태 업데이트 로직 ---

        # 규칙 6: 경로 복귀 확인
        # 경로 이탈 상태였는데, 원래 있어야 할 위치(path[path_cursor])로 돌아왔는지 확인
        if self.off_path and self.pos == self.path[self.path_cursor]:
            self.off_path = False
            # 복귀했으므로, 다음 스텝부터는 정상 경로를 따름
            self.update_control_buffer()
            return

        # 규칙 3 & 4: 경로 이탈 발생 및 유지 확인
        # 정상 주행 중이었어야 할 다음 위치
        expected_next_pos_on_path = self.path[self.path_cursor + 1]
        
        # 실제 이동한 위치가 예상 경로와 다른 경우
        if self.pos != expected_next_pos_on_path:
            # 교차로 등에 의해 강제로 다른 곳으로 이동된 경우
            self.off_path = True
        else:
            if self.path_cursor < len(self.path) - 1:
                self.path_cursor += 1
        
        # 다음 스텝에 사용할 제어 신호를 미리 계산
        self.update_control_buffer()
