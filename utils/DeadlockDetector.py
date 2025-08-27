class DeadlockDetector:
    """
    교차로 내 AMR들의 경로를 분석하여 잠재적인 데드락(Swapping Conflict)을
    예측하는 클래스.
    """
    def __init__(self, controller_ref):
        self.controller = controller_ref

    def check_deadlock(self, intersection):
        """
        '중앙 AMR(center_agv)이 관련된' 데드락만 True로 반환.
        중앙 AMR이 없거나, 교차로 내 AGV가 2대 미만이면 False.
        """
        center = intersection.center_agv
        if center is None:
            return False

        agvs_inside = intersection.agvs_in_intersection
        if len(agvs_inside) < 2:
            return False

        # 중앙 AMR과 나머지 AGV들만 검사
        for other in agvs_inside:
            if other is center:
                continue
            # 양방향(중앙→상대, 상대→중앙) 모두 확인
            if (self._check_swapping_path(center, other, intersection) or
                self._check_swapping_path(other, center, intersection)):
                return True

        return False

    def _check_swapping_path(self, agv1, agv2, intersection):
        """
        A(agv1)의 경로 상에 B(agv2)의 현재 위치가 포함되어 있고,
        A의 해당 구간 역순이 B의 경로에 서브시퀀스로 포함되면 스와핑 위험으로 판단.
        """
        path1 = self.controller.agv_path.get(agv1.id)
        path2 = self.controller.agv_path.get(agv2.id)
        if not path1 or not path2:
            return False

        pos2 = agv2.pos

        # A의 경로에서 B의 현재 위치 인덱스 찾기
        try:
            index2_in_1 = path1.index(pos2)
        except ValueError:
            return False

        # A의 경로 구간을 뒤집고, B의 경로에 포함되는지 확인
        sub_path1 = path1[:index2_in_1 + 1]
        if not sub_path1:
            return False
        reversed_sub_path1 = sub_path1[::-1]

        L = len(reversed_sub_path1)
        for i in range(len(path2) - L + 1):
            if path2[i:i + L] == reversed_sub_path1:
                return True
        return False
