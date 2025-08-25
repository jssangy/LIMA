from itertools import combinations

class DeadlockDetector:
    """
    교차로 내 AMR들의 경로를 분석하여 잠재적인 데드락(Swapping Conflict)을
    예측하는 클래스.
    """
    def __init__(self, controller_ref):
        self.controller = controller_ref

    def check_deadlock(self, intersection):
        """
        교차로 내 모든 AMR 쌍의 경로를 분석하여 데드락 발생 가능성을 확인.
        """
        agvs_inside = intersection.agvs_in_intersection

        # 교차로 내 AMR이 2대보다 적으면 데드락 검사 안 함
        if len(agvs_inside) < 2:
            return False

        # 모든 AMR 쌍에 대해 경로 엇갈림 검사
        for agv1, agv2 in combinations(agvs_inside, 2):
            if self._check_swapping_path(agv1, agv2, intersection) or \
                self._check_swapping_path(agv2, agv1, intersection):
                 return True
            
        return False
    
    def _check_swapping_path(self, agv1, agv2, intersection):
        """
        A의 경로에 B의 현재 위치가 있는지 확인하고, 경로를 뒤집어
        B의 경로에 포함되는지 검사합니다.
        """
        # 1. 정보 가져오기
        path1 = self.controller.agv_path[agv1.id]
        pos2 = agv2.pos
        path2 = self.controller.agv_path[agv2.id]

        # 2. A의 경로에서 B의 현재 위치 인덱스 찾기
        try:
            index2_in_1 = path1.index(pos2)
        except ValueError:
            # A의 경로에 B의 위치가 없으면 충돌이 아님
            return False

        # 3. 경로를 슬라이싱하고 뒤집기
        sub_path1 = path1[:index2_in_1 + 1]
        reversed_sub_path1 = sub_path1[::-1]

        # 4. B의 경로에서 A의 역방향 경로가 포함되어 있는지 확인
        return any(reversed_sub_path1 == path2[i:i + len(reversed_sub_path1)]
                   for i in range(len(path2) - len(reversed_sub_path1) + 1))