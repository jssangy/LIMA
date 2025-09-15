import colorsys
import random

class Color_dict():
    def __init__(self, agv_num):
        self.dic = {}
        
        # 황금비의 역수 (conjugate)
        # 이 값을 계속 더해주면 색상환을 가장 균일하게 채울 수 있습니다.
        golden_ratio_conjugate = 0.61803398875 
        
        # 시작 색상을 무작위로 선택하여 매번 다른 색상 세트를 생성
        h = random.random() 
        
        for i in range(agv_num):
            # 채도(saturation)와 명도(value)는 높게 유지하여 밝은 색상을 보장합니다.
            s = 0.9
            v = 1.0
            
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            rgb_255 = (int(r * 255), int(g * 255), int(b * 255))
            self.dic[i] = rgb_255
            
            # 다음 색상은 황금비만큼 떨어진 위치로 이동
            h = (h + golden_ratio_conjugate) % 1.0