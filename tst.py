# === quick_sim_plan_action.py ===
from copy import deepcopy
import time

# 1) 프로젝트의 Intersection import (경로는 네 프로젝트에 맞게 수정)
# from utils.Intersection import Intersection
from utils.Intersection import Intersection  # 예시: 같은 폴더/패키지에 있을 때

CAP = 5  # 모든 팔 길이 5

# ------------------------------------------------------------
# 간편 스펙 -> (init_stacks, true_target) 변환기
# ------------------------------------------------------------
def make_case_from_goal_spec(goal_spec, cap=CAP, start_id=1):
    """
    goal_spec 예:
      {"N": ["S","E","W","N"], "E": ["N","W","E"], "S": "NW", "W": "EN"}
    반환:
      init_stacks: {"N":[1,2,...], ...}  # TOP=0번
      true_target: {1:"S", 2:"E", ...}   # ID -> 목표팔
    """
    order = ["N","E","S","W"]

    def norm(goals):
        # 리스트/튜플: 원소들을 대문자 문자열로
        if isinstance(goals, (list, tuple)):
            return [str(x).strip().upper() for x in goals if str(x).strip()]
        # 문자열: "SEWN" 또는 "S E W N" 또는 "S,E,W,N" 모두 허용
        if isinstance(goals, str):
            s = goals.strip().upper()
            # 공백/콤마 기준 토큰화
            tokens = [t for t in s.replace(",", " ").split() if t]
            # 토큰이 하나고 길이가 2이상이면 문자 분해("SEWN" -> ["S","E","W","N"])
            if len(tokens) == 1 and len(tokens[0]) > 1:
                tokens = list(tokens[0])
            return tokens
        raise ValueError(f"Unsupported goals type: {type(goals)}")

    # 유효성 및 변환
    init_stacks = {d: [] for d in order if d in goal_spec}
    true_target = {}
    next_id = start_id

    for arm in order:
        if arm not in goal_spec:
            continue
        goals = norm(goal_spec[arm])
        if any(g not in order for g in goals):
            bad = [g for g in goals if g not in order]
            raise ValueError(f"Invalid goal(s) {bad} in arm {arm}. Use only N/E/S/W.")
        if len(goals) > cap:
            raise ValueError(f"Arm {arm} has {len(goals)} items but cap is {cap}.")
        # 입력 순서 그대로(=TOP이 0번)
        for g in goals:
            init_stacks[arm].append(next_id)
            true_target[next_id] = g
            next_id += 1

    return init_stacks, true_target

# ------------------------------------------------------------
# 여기만 간단히 적으면 됩니다 👇
#   - TOP은 리스트/문자열의 맨 앞입니다.
# ------------------------------------------------------------
goal_spec = {
    "N": "SEW",   
    "E": "WSNN",
    "S": "NWEN",       
    "W": "ENSN",
}

# 2) Intersection 인스턴스 (팔 길이 5, present_dirs는 스펙 키에서 자동)
I = Intersection(
    intersection_data=(10, 10, CAP, CAP, CAP, CAP),
    neighbors_map={},
    present_dirs=set(goal_spec.keys()),
)

# 3) 간편 스펙을 실제 init_stacks/true_target으로 확장
init_stacks, true_target = make_case_from_goal_spec(goal_spec, cap=CAP)

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
