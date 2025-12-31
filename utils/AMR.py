from typing import Tuple

# AMR Object
class AMR():
    def __init__(self, id, pos, goal, color):
        # Basic information
        self.id = id
        self.color = color
        self.goal = goal
        self.pos = pos
        self.prev_pos = pos
        self.next_pos = pos
        self.steps = 0       
        self.prev_moved = False
        self.no_move_steps = 0

        # Path
        self.path = []
        self.path_cursor = 0
        self.scheduling = 0

        self.path_remaining: set[Tuple[int, int]] = set()
        self.path_orig_len = 0


    def reset(self):
        """
        Resets the state of the AMR.
        """
        self.prev_pos = self.pos
        self.next_pos = self.pos
        self.steps = 0
        self.prev_moved = False
        self.no_move_steps = 0

        self.path = []
        self.path_cursor = 0
        self.scheduling = 0
        self.path_remaining: set[Tuple[int, int]] = set()
        self.path_orig_len = 0


    def set_path(self, new_path: list):
        """
        Rule 1: Receives a new full path from the planner and resets the state.
        """
        if not new_path:
            self.path = [self.pos] # If there is no path, set it to stay in place
        else:
            self.path = new_path

        self.path_cursor = 0
        self.next_pos = self.path[1] if len(self.path) > 1 else self.pos
        self.path_remaining.discard(tuple(self.pos))

    
    def path_integrity_ratio(self):
        if self.path_orig_len == 0:
            return 1.0
        
        return (self.path_orig_len - self.scheduling) / self.path_orig_len * 100.0


    def move(self):
        """
        After moving, find the current position on the path and synchronize path_cursor.
        """
        self.prev_pos = self.pos
        self.path_cursor += 1
        self.pos = self.next_pos
        self.next_pos = self.path[self.path_cursor + 1] if self.path_cursor + 1 < len(self.path) else self.pos
        self.steps += 1
        self.prev_moved = True
        self.no_move_steps = 0
        if self.scheduling > 0:
            self.scheduling -= 1
        if self.path_remaining:
            self.path_remaining.discard(self.pos)