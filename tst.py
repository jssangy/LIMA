# === quick_sim_plan_action.py ===
from copy import deepcopy

# 1) 프로젝트의 Intersection import (경로는 네 프로젝트에 맞게 수정)
# from utils.Intersection import Intersection
from utils.Intersection import Intersection  # 예시: 같은 폴더/패키지에 있을 때

# 2) 테스트용 Intersection 인스턴스 (팔 길이는 초기 스택보다 넉넉하게!)
I = Intersection(
    intersection_data=(10, 10, 5, 5, 5, 5),   # (cx, cy, len_N, len_E, len_S, len_W)
    neighbors_map={},
    present_dirs={'N', 'E', 'S', 'W'},
)

# 3) 초기 스택 상태 정의: TOP이 리스트의 0번 입니다!
#    예: N에 [1,2]면 1이 TOP(센터에서 가장 가까움), 2가 그 뒤.
init_stacks = {
        'N': [1, 2, 3],  # 1->N(정답), 2->E(이질), 3->N
        'E': [4],        # -> W
        'S': [5],        # -> N
        'W': [6],        # -> S
}

# 4) 아이템별 '진짜 목표' 매핑
true_target = {
        1:'N', 2:'E', 3:'N', 4:'W', 5:'N', 6:'S'
}

# (참고) plan_action은 우리가 아래에서 몽키패치할 build_stacks_from_snapshot()의
# 반환값을 사용하므로 amr_intent_map을 채우지 않아도 됩니다.
# 필요하면 다음과 같이 채워도 무방합니다(사용되지는 않음).
I.amr_intent_map = {
    aid: {'amr_obj': None, 'current_arm': arm, 'exit_arm': true_target[aid]}
    for arm, ids in init_stacks.items() for aid in ids
}

# 5) build_stacks_from_snapshot 몽키패치:
#    새 구현은 (stacks, targets) 튜플을 반환해야 합니다.
orig_bss = I.build_stacks_from_snapshot
def _patched_bss():
    return (deepcopy(init_stacks), deepcopy(true_target))
I.build_stacks_from_snapshot = _patched_bss

# 6) plan_action 실행
actions = I.plan_action()

# 7) 몽키패치 복원
I.build_stacks_from_snapshot = orig_bss

# 8) 액션 재적용 시뮬레이션(검증)
#    TOP pop(0) → dst insert(0) 규칙을 그대로 적용
cap = {'N': I.len_N, 'E': I.len_E, 'S': I.len_S, 'W': I.len_W}
final_stacks = deepcopy(init_stacks)

def apply_action(stacks, src, dst):
    # 간단한 용량/빈 스택 체크
    if not stacks[src]:
        print(f"[WARN] empty pop on {src} -> {dst}")
        return False
    if len(stacks[dst]) >= cap[dst]:
        print(f"[WARN] capacity overflow on {src} -> {dst}")
        return False
    aid = stacks[src].pop(0)        # src TOP pop
    stacks[dst].insert(0, aid)      # dst TOP push
    return True

def stacks_are_pure(stacks, targets) -> bool:
    for arm, ids in stacks.items():
        if any(targets[aid] != arm for aid in ids):
            return False
    return True

def pretty(stacks):
    return " | ".join(f"{d}:{ids}" for d, ids in stacks.items())

ok = True
for (src, dst) in actions:
    if not apply_action(final_stacks, src, dst):
        ok = False

# 9) 결과 출력 및 단언
print("\n=== Simulation Result ===")
print(f"Actions ({len(actions)} moves): {actions[:15]}{' ...' if len(actions)>15 else ''}")
print("Final stacks:", pretty(final_stacks))
print("Capacities :", cap)

assert ok, "액션 적용 중 용량/빈 스택 위반이 발생했어요."
assert stacks_are_pure(final_stacks, true_target), "최종 스택이 목표 기준으로 순수하지 않습니다."

print("✅ Test passed: all stacks are pure by target and within capacity.")
