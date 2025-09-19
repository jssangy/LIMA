from typing import List, Tuple

# AGV Object
class agv():
    def __init__(self, pos, id, color):
        # ID of AGV
        self.id = id

        # Color of AGV
        self.color = color
        
        # current position of agv
        self.pos = pos

        # previous position of agv
        self.prev_pos = pos

        # goal position
        self.goal = (0, 0)

        self.steps = 0
        
        self.action_count = 0

        self.path_remaining: set[Tuple[int,int]] = set()
        self.path_orig_len = 0

    def set_initial_path(self, path: List[Tuple[int,int]]):
        # tuple(int,int)로 정규화 + 유니크 집합 생성
        s = {(int(x), int(y)) for (x, y) in path}
        self.path_orig_len = len(s)
        self.path_remaining = set(s)
        # 시작 칸은 이미 밟고 시작하므로 제거
        self.path_remaining.discard(tuple(self.pos))

    def path_integrity_ratio(self) -> float:
        # 0.0 ~ 1.0
        if self.path_orig_len == 0: 
            return 1.0
        covered = self.path_orig_len - len(self.path_remaining)
        return covered / self.path_orig_len * 100.0

    def move(self, control_signal):
        self.prev_pos = self.pos
        self.pos = (self.pos[0] + control_signal[0], self.pos[1] + control_signal[1])
        self.steps += 1
        # 현재 위치를 경로 잔여에서 제거 (O(1))
        if self.path_remaining:
            self.path_remaining.discard(self.pos)