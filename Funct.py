class Color_dict():
    
    def __init__(self, agv_num):
        base_colors = [
            (255, 0, 0),       # A 빨강
            (0, 255, 0),       # B 초록
            (0, 0, 255),       # C 파랑
            (255, 255, 0),     # D 노랑
            (255, 0, 255),     # E 핑크
            (0, 255, 255),     # F 하늘
            (255, 165, 0),     # G 주황
            (128, 0, 128),     # H 보라
            (0, 128, 128),     # I 청록
            (128, 128, 0),     # J 올리브
            (255, 105, 180),   # K 핫핑크
            (0, 100, 0),       # L 진녹
            (139, 69, 19),     # M 갈색
            (70, 130, 180),    # N 강철파랑
            (210, 105, 30),    # O 초콜릿
        ]

        self.dic = {}

        for i in range(min(agv_num, 15)):  # 최대 15(A-O)개 할당 가능
            self.dic[chr(i + 65)] = base_colors[i]



def get_distance(pos1, pos2):
    x = abs(pos1[0] - pos2[0])
    y = abs(pos1[1] - pos2[1])
    if x == 0:
        return y
    if y == 0:
        return x