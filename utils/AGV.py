# AGV Object
class agv():
    turns = {}
    
    # pos is given as coordinates on the grid ex (1,5)
    def __init__(self, pos, id, color):
        # ID of AGV
        self.id = id

        # Color of AGV
        self.color = color

        # set start position      
        self.start = pos
        
        # current position of agv
        self.pos = pos

        # previous position of agv
        self.prev_pos = pos

        # goal position
        self.goal = (0, 0)
        
        # Current control state
        self.move_x = 0
        self.move_y = 0
        
        # 0: Normal, 1: Wall Collision, 2: AGV Collision (Deadlock)
        self.mode = 0
        
    def get_control(self, packet):
        self.move_x = packet[0][0]
        self.move_y = packet[0][1]
        self.mode = packet[1]
        
    def next_pos(self):
        return (self.pos[0] + self.move_x, self.pos[1] + self.move_y)

    def move(self, control_signal):
        self.prev_pos = self.pos
        self.pos = (self.pos[0] + control_signal[0], self.pos[1] + control_signal[1])

    # Send position and state
    def sensing(self):
        return [self.pos, self.mode]
