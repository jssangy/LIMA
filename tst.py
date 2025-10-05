# === quick_sim_plan_action.py ===
from copy import deepcopy

# 1) 프로젝트의 Intersection import (경로는 네 프로젝트에 맞게 수정)
# from utils.Intersection import Intersection
from utils.Intersection import Intersection  # 예시: 같은 폴더에 있을 때

# 3) 테스트용 Intersection 인스턴스 (팔 길이/방향은 자유롭게 조절)
I = Intersection(
    intersection_data=(10, 10, 3, 3, 3, 3),   # (cx, cy, len_N, len_E, len_S, len_W)
    neighbors_map={},
    present_dirs={'N', 'E', 'S', 'W'},
)

# 4) 초기 스택 상태 정의: TOP이 리스트의 0번 입니다!
#    예: N에 [1,2]면 1이 TOP(센터에서 가장 가까움), 2가 그 뒤.
init_stacks = {
    'N': [1, 2],
    'E': [3],
    'S': [4, 5],
    'W': [],
}
# 5) 진짜 목표(true_target) 정의
true_target = {
    1: 'E',
    2: 'E',
    3: 'S',
    4: 'N',
    5: 'W',
}

# 6) intent 맵 채우기(현재 엣지/탈출 엣지 정보) — plan_action 내부 일부 분기에서 참조
I.amr_intent_map = {}
for d, ids in init_stacks.items():
    for aid in ids:
        I.amr_intent_map[aid] = {
            'amr_obj': None,          # plan_action에는 필요 없음
            'current_arm': d,         # 현재 엣지
            'exit_arm': true_target[aid],  # 탈출 엣지
        }

# 7) build_stacks_from_snapshot 몽키패치: 이번 1회만 우리가 지정한 상태를 반환
orig_bss = I.build_stacks_from_snapshot
def _patched_bss():
    return (deepcopy(init_stacks), deepcopy(true_target))
I.build_stacks_from_snapshot = _patched_bss

# 8) 실행
actions = I.plan_action()

# 9) 몽키패치 복원
I.build_stacks_from_snapshot = orig_bss

# 10) 액션 재적용 시뮬레이션(검증)
#     TOP pop(0) → dst insert(0) 규칙을 그대로 적용
cap = {'N': I.len_N, 'E': I.len_E, 'S': I.len_S, 'W': I.len_W}
final_stacks = deepcopy(init_stacks)

def apply_action(stacks, src, dst):
    if not stacks[src]:
        return False
    if len(stacks[dst]) >= cap[dst]:
        return False
    aid = stacks[src].pop(0)
    stacks[dst].insert(0, aid)
    return True

ok = True
for (src, dst) in actions:
    if not apply_action(final_stacks, src, dst):
        ok = False
        print(f"[WARN] capacity/empty violation on action {(src, dst)}; skipped.")


