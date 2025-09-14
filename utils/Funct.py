import colorsys
import random

class Color_dict():
    def __init__(self, agv_num):
        self.dic = {}
        for i in range(agv_num):
            # 결정적인 색상 대신 랜덤한 색조(hue)를 생성합니다.
            h = random.random()
            # 채도(saturation)와 명도(value)는 높게 유지하여 밝은 색상을 보장합니다.
            s = 0.9
            v = 1.0
            
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            rgb_255 = (int(r * 255), int(g * 255), int(b * 255))
            self.dic[i] = rgb_255