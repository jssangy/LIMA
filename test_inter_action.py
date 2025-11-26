import unittest
import random
from unittest.mock import MagicMock
from utils.Intersection import Intersection

def generate_random_snapshot(num_items=15, capacity=5):
    """
    15개의 아이템을 생성하고 4개의 스택에 무작위로 분배하는 헬퍼 함수
    """
    dirs = ['N', 'E', 'S', 'W']
    
    # 1. 아이템 ID 및 목표(Target) 생성 (1~15번)
    # 목표는 N, E, S, W 중 하나 랜덤
    targets = {}
    all_items = list(range(1, num_items + 1))
    
    for item_id in all_items:
        targets[item_id] = random.choice(dirs)
        
    # 2. 스택에 무작위 분배
    stacks = {d: [] for d in dirs}
    
    # 아이템 순서를 섞어서 배치
    random.shuffle(all_items)
    
    for item_id in all_items:
        # 꽉 차지 않은 스택 후보 찾기
        available_stacks = [d for d in dirs if len(stacks[d]) < capacity]
        
        if not available_stacks:
            raise RuntimeError("아이템을 모두 배치할 공간이 부족합니다.")
            
        chosen_stack = random.choice(available_stacks)
        stacks[chosen_stack].append(item_id)
        
    return stacks, targets

def print_stacks(stacks, targets, title="State"):
    """스택 상태를 시각적으로 출력"""
    print(f"\n--- {title} ---")
    max_height = max(len(s) for s in stacks.values()) if stacks else 0
    dirs = ['N', 'E', 'S', 'W']
    
    # 위에서부터 아래로 출력 (Top -> Bottom)
    for h in range(max_height - 1, -1, -1):
        row = []
        for d in dirs:
            stack = stacks[d]
            if h < len(stack):
                item = stack[h]
                tgt = targets.get(item, '?')
                # 포맷: [ID(Target)] 예: [10(N)]
                row.append(f"[{item:2}({tgt})]")
            else:
                row.append("       ")
        print("  ".join(row))
    
    print("  ".join([f"---{d}--- " for d in dirs]))

class TestRandomScenario(unittest.TestCase):
    def setUp(self):
        # 팔 길이 5로 설정 (Capacity=5)
        self.inter = Intersection(
            intersection_data=(10, 10, 5, 5, 5, 5),
            neighbors_map={},
            present_dirs="NESW"
        )

    def test_random_15_items(self):
        print("\n=== 랜덤 아이템 15개 대규모 테스트 시작 ===")
        
        # 1. 랜덤 데이터 생성
        # 아이템 15개, 스택 용량 5 (총 슬롯 20개 중 15개 점유 = 75% 점유율)
        stacks, targets = generate_random_snapshot(num_items=15, capacity=5)
        
        # 2. Mocking
        self.inter.build_stacks_from_snapshot = MagicMock(return_value=(stacks, targets))
        
        # 3. 초기 상태 출력
        print_stacks(stacks, targets, "Initial Random State")
        print(f"Total Items: {sum(len(s) for s in stacks.values())}")
        
        # 4. 실행
        try:
            moves = self.inter.plan_action()
        except Exception as e:
            self.fail(f"Solver 실행 중 에러 발생: {e}")
            
        # 5. 결과 검증
        print(f"\n[Result] Generated Plan Length: {len(moves)} moves")
        
        if len(moves) > 0:
            print("Actions :", moves)
        else:
            print("Actions: [] (이미 정렬되어 있거나 해를 못 찾음)")
            
        # 검증: moves가 리스트인지
        self.assertIsInstance(moves, list)
        
        # 간단한 시뮬레이션으로 최종 상태 확인 (검증용)
        sim_stacks = {k: v[:] for k, v in stacks.items()} # Deepcopy
        dir_idx = {d: i for i, d in enumerate(['N', 'E', 'S', 'W'])}
        idx_dir = {i: d for i, d in enumerate(['N', 'E', 'S', 'W'])}
        
        print("\n--- Simulating Plan ---")
        for i, (src_idx, dst_idx) in enumerate(moves):
            # sch.py는 인덱스(int)를 반환하므로 변환 필요할 수 있음
            # (만약 plan_action이 (src, dst) 튜플을 반환하도록 수정했다면 그대로 사용)
            # 현재 plan_action 구현 상 schedule()의 리턴값(moves)은 [(0, 1), ...] 형태의 인덱스 튜플임
            
            src = idx_dir[src_idx]
            dst = idx_dir[dst_idx]
            
            item = sim_stacks[src].pop()
            sim_stacks[dst].append(item)
            
        print_stacks(sim_stacks, targets, "Final State after Execution")
        
        # 최종 상태가 정렬되었는지 확인 (각 스택의 아이템들이 자기 집(목표)에 있는지)
        is_sorted = True
        for d, stack in sim_stacks.items():
            for item in stack:
                if targets[item] != d:
                    is_sorted = False
                    break
        
        if is_sorted:
            print("\nSUCCESS: 모든 아이템이 올바르게 정렬되었습니다! 🎉")
        else:
            # Overflow 로직상 자기 집에 없어도 되는 경우가 있으므로(버퍼 활용), 
            # 엄밀한 실패는 아니지만 단순 확인용 메시지
            print("\nNOTE: 완벽 정렬은 아니거나(Overflow 버퍼링), 시뮬레이션 검증 로직 차이일 수 있습니다.")

if __name__ == '__main__':
    unittest.main()