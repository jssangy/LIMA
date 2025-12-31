import colorsys

_PHI_CONJ = 0.6180339887498948  # Golden ratio conjugate (evenly distributes hue)

def _id_to_color(key, s: float = 0.85, v: float = 0.95):
    """AGV id -> RGB(0~255). Stable mapping even if not an integer."""
    try:
        k = int(key)
    except Exception:
        k = abs(hash(key)) & 0xFFFFFFFF
    h = (k * _PHI_CONJ) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


class _AutoColorDict(dict):
    """A dict that immediately generates and fills a color when a non-existent key is queried."""
    def __init__(self, s: float = 0.85, v: float = 0.95):
        super().__init__()
        self._s, self._v = s, v

    def __missing__(self, key):
        color = _id_to_color(key, self._s, self._v)
        self[key] = color
        return color

    # Handle automatic generation even with .get() (backward compatibility)
    def get(self, key, default=None):
        return self[key]


class Color_dict:
    def __init__(self, agv_num: int = 0, s: float = 0.85, v: float = 0.95, prefill: bool = True):
        """
        agv_num: Initial N items can be pre-filled (compatibility); IDs beyond that are automatically generated.
        """
        self.dic = _AutoColorDict(s=s, v=v)
        if prefill and agv_num > 0:
            for i in range(agv_num):
                _ = self.dic[i]  # Initial pre-fill

def get_distance(pos1, pos2):
    x = abs(pos1[0] - pos2[0])
    y = abs(pos1[1] - pos2[1])
    if x == 0:
        return y
    if y == 0:
        return x
    return x + y  # ← If both axes differ, come here
