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

    def move(self, control_signal):
        self.prev_pos = self.pos
        self.pos = (self.pos[0] + control_signal[0], self.pos[1] + control_signal[1])
