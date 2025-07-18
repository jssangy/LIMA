import colorsys

class Color_dict():
    def __init__(self, agv_num):
        self.dic = {}
        for i in range(agv_num):
            h = i / agv_num
            r, g, b = colorsys.hsv_to_rgb(h, 1, 1)
            rgb_255 = (int(r * 255), int(g * 255), int(b * 255))
            self.dic[str(i)] = rgb_255



def get_distance(pos1, pos2):
    x = abs(pos1[0] - pos2[0])
    y = abs(pos1[1] - pos2[1])
    if x == 0:
        return y
    if y == 0:
        return x