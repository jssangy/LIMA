import colorsys

_PHI_CONJ = 0.6180339887498948  # 황금비 켤레 (고르게 hue 분산)

def _id_to_color(key, s: float = 0.85, v: float = 0.95):
    """AGV id -> RGB(0~255). 정수가 아니어도 안정적으로 매핑."""
    try:
        k = int(key)
    except Exception:
        k = abs(hash(key)) & 0xFFFFFFFF
    h = (k * _PHI_CONJ) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


class _AutoColorDict(dict):
    """존재하지 않는 키를 조회하면 즉시 색을 생성해 채우는 dict."""
    def __init__(self, s: float = 0.85, v: float = 0.95):
        super().__init__()
        self._s, self._v = s, v

    def __missing__(self, key):
        color = _id_to_color(key, self._s, self._v)
        self[key] = color
        return color

    # .get()으로도 자동 생성되게 처리 (기존 코드 호환)
    def get(self, key, default=None):
        return self[key]


class Color_dict:
    def __init__(self, agv_num: int = 0, s: float = 0.85, v: float = 0.95, prefill: bool = True):
        """
        agv_num: 초기 N개만 미리 채워둘 수 있음(호환성); 그 이상 id는 자동 생성.
        """
        self.dic = _AutoColorDict(s=s, v=v)
        if prefill and agv_num > 0:
            for i in range(agv_num):
                _ = self.dic[i]  # 초기 프리필

def get_distance(pos1, pos2):
    x = abs(pos1[0] - pos2[0])
    y = abs(pos1[1] - pos2[1])
    if x == 0:
        return y
    if y == 0:
        return x
    return x + y  # ← 두 축 모두 차이 나면 여기로
