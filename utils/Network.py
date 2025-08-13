import random

class network():
    def __init__(self):
        pass
    
    # Network
    def send(self, packet, loss_rate = 0):
        if random.random() > loss_rate:
            return packet
        else:
            return None