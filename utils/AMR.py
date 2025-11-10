from typing import Tuple

# AMR Object
class AMR():
    def __init__(self, id, pos, goal, color):
        # 기본 정보
        self.id = id
        self.color = color
        self.goal = goal
        self.pos = pos
        self.prev_pos = pos
        self.next_pos = pos
        self.steps = 0        

        # 자율 주행 및 경로 복귀를 위한 상태
        self.path = []
        self.path_cursor = 0
        self.scheduling = 0

        self.path_remaining: set[Tuple[int, int]] = set()
        self.path_orig_len = 0


    def reset(self):
        """
        AMR의 상태를 초기화합니다.
        """
        self.prev_pos = self.pos
        self.next_pos = self.pos
        self.steps = 0
        self.path = []
        self.path_cursor = 0
        self.scheduling = 0


    def set_path(self, new_path: list):
        """
        규칙 1: 플래너로부터 새로운 전체 경로를 주입받고 상태를 초기화합니다.
        """
        if not new_path:
            self.path = [self.pos] # 경로가 없으면 제자리에 머무는 경로로 설정
        else:
            self.path = new_path

        self.path_cursor = 0
        self.next_pos = self.path[1] if len(self.path) > 1 else self.pos
        self.path_remaining.discard(tuple(self.pos))

    
    def insert_scheduled_path(self, new_path: list):
        if not new_path:
            return
        
        cur_idx = self.path_cursor
        prefix = self.path[:cur_idx + 1]

        to_insert = new_path[1:]

        center_cell = to_insert[-1]
        center_idx = None
        for j in range(cur_idx, len(self.path)):
            if self.path[j] == center_cell:
                center_idx = j
                break

        suffix = self.path[center_idx + 1:]
        self.path = prefix + to_insert + suffix
        self.scheduling = len(to_insert)

    
    def path_integrity_ratio(self):
        if self.path_orig_len == 0:
            return 1.0
        
        return (self.path_orig_len - self.scheduling) / self.path_orig_len * 100.0


    def move(self, freeze=False):
        """
        [수정] 이동 후, 경로상의 현재 위치를 찾아 path_cursor를 동기화합니다.
        """
        if freeze:
            return

        self.prev_pos = self.pos
        self.path_cursor += 1
        self.pos = self.path[self.path_cursor]
        self.next_pos = self.path[self.path_cursor + 1] if self.path_cursor + 1 < len(self.path) else self.pos
        self.steps += 1
        if self.scheduling > 0:
            self.scheduling -= 1
        if self.path_remaining:
            self.path_remaining.discard(self.pos)