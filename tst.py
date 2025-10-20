# === quick_sim_plan_action.py ===
from copy import deepcopy
import time

# 1) 프로젝트의 Intersection import (경로는 네 프로젝트에 맞게 수정)
# from utils.Intersection import Intersection
from utils.Intersection import Intersection  # 예시: 같은 폴더/패키지에 있을 때

CAP = 5  # 모든 팔 길이 5

# ------------------------------------------------------------
# 간편 스펙 -> (init_stacks, true_target) 변환기
#   * 입력 규칙: goal_spec에서 "맨 앞 토큰"이 TOP(센터 가까운 쪽)
#   * 플래너는 오른쪽=TOP이므로, 생성 후 리스트를 뒤집어 오른쪽이 TOP이 되도록 맞춤
# ------------------------------------------------------------
def make_case_from_goal_spec(goal_spec, cap=CAP, start_id=1):
    order = ["N","E","S","W"]

    def norm(goals):
        if isinstance(goals, (list, tuple)):
            return [str(x).strip().upper() for x in goals if str(x).strip()]
        if isinstance(goals, str):
            s = goals.strip().upper()
            tokens = [t for t in s.replace(",", " ").split() if t]
            if len(tokens) == 1 and len(tokens[0]) > 1:
                tokens = list(tokens[0])
            return tokens
        raise ValueError(f"Unsupported goals type: {type(goals)}")

    init_stacks = {d: [] for d in order if d in goal_spec}
    true_target = {}
    next_id =  start_id

    for arm in order:
        if arm not in goal_spec:
            continue
        goals = norm(goal_spec[arm])
        if any(g not in order for g in goals):
            bad = [g for g in goals if g not in order]
            raise ValueError(f"Invalid goal(s) {bad} in arm {arm}. Use only N/E/S/W.")
        if len(goals) > cap:
            raise ValueError(f"Arm {arm} has {len(goals)} items but cap is {cap}.")

        ids = []
        for g in goals:          # 입력: 맨 앞이 TOP(near)
            ids.append(next_id)
            true_target[next_id] = g
            next_id += 1

        # ✅ 플래너는 오른쪽=TOP이므로, 리스트를 뒤집어서 오른쪽이 TOP이 되게 맞춘다.
        init_stacks[arm] = ids[::-1]

    return init_stacks, true_target

# ------------------------------------------------------------
# 여기만 간단히 적으면 됩니다 👇
#   - goal_spec 입력은 "맨 앞이 TOP"
# ------------------------------------------------------------
goal_spec = {
    "N": "SEW",
    "E": "WSNN",
    "S": "NWEN",
    "W": "ENSN",
}

# 2) Intersection 인스턴스 (팔 길이 5)
I = Intersection(
    intersection_data=(10, 10, CAP, CAP, CAP, CAP),
    neighbors_map={},
    present_dirs=set(goal_spec.keys()),
)

# 3) 스펙을 실제 init_stacks/true_target으로 확장 (오른쪽=TOP 형태로 반환됨)
init_stacks, true_target = make_case_from_goal_spec(goal_spec, cap=CAP)

# (참고) plan_action은 아래에서 몽키패치된 build_stacks_from_snapshot()의
# 반환값을 사용하므로 amr_intent_map은 필수 아님.
I.amr_intent_map = {
    aid: {'amr_obj': None, 'current_arm': arm, 'exit_arm': true_target[aid]}
    for arm, ids in init_stacks.items() for aid in ids
}

# 5) build_stacks_from_snapshot 몽키패치: (stacks, targets) 반환
orig_bss = I.build_stacks_from_snapshot
def _patched_bss():
    return (deepcopy(init_stacks), deepcopy(true_target))
I.build_stacks_from_snapshot = _patched_bss

# 6) plan_action 실행
actions = I.plan_action()

# 7) 몽키패치 복원
I.build_stacks_from_snapshot = orig_bss

# 8) 액션 재적용 시뮬레이션(검증)
#    ✅ 오른쪽=TOP 규칙으로 적용: src.pop() → dst.append()
cap = {'N': I.len_N, 'E': I.len_E, 'S': I.len_S, 'W': I.len_W}
final_stacks = deepcopy(init_stacks)

def apply_action(stacks, src, dst):
    if not stacks[src]:
        print(f"[WARN] empty pop on {src} -> {dst}")
        return False
    if len(stacks[dst]) >= cap[dst]:
        print(f"[WARN] capacity overflow on {src} -> {dst}")
        return False
    aid = stacks[src].pop()     # ✅ 오른쪽에서 pop (TOP)
    stacks[dst].append(aid)     # ✅ 오른쪽에 push (TOP)
    return True

def stacks_are_pure(stacks, targets) -> bool:
    for arm, ids in stacks.items():
        if any(targets[aid] != arm for aid in ids):
            return False
    return True

def pretty(stacks):
    return " | ".join(f"{d}:{ids}" for d, ids in stacks.items())

ok = True
start_time = time.time()
for (src, dst) in actions:
    if not apply_action(final_stacks, src, dst):
        ok = False
end = time.time()

# 9) 결과 출력 및 단언
print(f"\n=== Simulation Result (Time: {end - start_time}s) ====")
print(f"Actions ({len(actions)} moves): {actions[:15]}{' ...' if len(actions)>15 else ''}")
print("Final stacks:", pretty(final_stacks))
print("Capacities :", cap)

assert ok, "액션 적용 중 용량/빈 스택 위반이 발생했어요."
assert stacks_are_pure(final_stacks, true_target), "최종 스택이 목표 기준으로 순수하지 않습니다."
print("✅ Test passed: all stacks are pure by target and within capacity.")
