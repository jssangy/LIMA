NUM_STACKS = 4          # 스택 개수
STACK_CAPACITY = 5      # 각 스택이 담을 수 있는 최대 원소 수

from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional
import random

@dataclass
class StackRearrangementEnv:
    def __init__(self, num_stacks, stack_capacity, stacks):
        self.num_stacks = num_stacks
        self.stack_capacity = stack_capacity
        self.stacks = stacks

    def __post_init__(self) -> None:
        if not self.stacks:
            self.reset()

    def reset(self, seed: Optional[int] = None) -> None:
        rng = random.Random(seed)
        total_items = max(0, (self.num_stacks - 1) * self.stack_capacity)
        
        # 각 원소 종류별로 무작위 개수 생성 (개수 제한 없음)
        # 전체 합이 total_items가 되도록 보장
        if total_items == 0:
            self.stacks = [[] for _ in range(self.num_stacks)]
            return
        
        # 각 색상에 대해 무작위 가중치 생성
        weights = [rng.random() for _ in range(self.num_stacks)]
        total_weight = sum(weights)
        
        # 가중치를 정규화하여 각 색상의 개수 결정
        counts = []
        allocated = 0
        for i in range(self.num_stacks):
            if i == self.num_stacks - 1:
                # 마지막 색상은 나머지 모두 할당 (합 보장)
                count = total_items - allocated
            else:
                count = int(weights[i] / total_weight * total_items)
                allocated += count
            counts.append(max(0, count))
        
        # 각 색상의 원소 리스트 생성
        items = []
        for color in range(self.num_stacks):
            items.extend([color] * counts[color])
        
        # 원소들을 무작위로 섞음
        rng.shuffle(items)
        
        # 스택에 무작위로 배치
        self.stacks = [[] for _ in range(self.num_stacks)]
        available = [self.stack_capacity] * self.num_stacks
        for item in items:
            candidates = [i for i, cap in enumerate(available) if cap > 0]
            if not candidates:
                break
            slot = rng.choice(candidates)
            self.stacks[slot].append(item)
            available[slot] -= 1

    def peek(self, stack_id: int) -> Optional[int]:
        stack = self.stacks[stack_id]
        return stack[-1] if stack else None

    def pop(self, stack_id: int) -> Optional[int]:
        if not self.stacks[stack_id]:
            return None
        return self.stacks[stack_id].pop()

    def push(self, stack_id: int, item: int) -> bool:
        if len(self.stacks[stack_id]) >= self.stack_capacity:
            return False
        self.stacks[stack_id].append(item)
        return True

    def move(self, src: int, dst: int) -> bool:
        if src == dst or not self.stacks[src]:
            return False
        if len(self.stacks[dst]) >= self.stack_capacity:
            return False
        item = self.stacks[src].pop()
        self.stacks[dst].append(item)
        return True

    def is_goal(self, goal_state: List[List[int]]) -> bool:
        return self.stacks == goal_state

    def is_solved(self) -> bool:
        # 기본 조건: 모든 스택이 자기 색상만 포함
        if all(all(item == stack_id for item in stack) for stack_id, stack in enumerate(self.stacks)):
            return True

        # Overflow 색상 확인: 특정 색상의 원소 수가 stack_capacity보다 많은 경우
        counts = Counter(item for stack in self.stacks for item in stack)
        overflow_types = set(
            item_type
            for item_type, total in counts.items()
            if total > self.stack_capacity
        )

        # Overflow 색상이 있으면 특별한 규칙 적용 (모든 overflow 색상을 동시에 고려)
        if overflow_types:
            return self._is_valid_with_overflow(overflow_types)

        return False

    def _is_valid_with_overflow(self, overflow_types: set[int]) -> bool:
        """여러 overflow 색상이 있을 때 해결 조건 확인."""
        for stack_id, stack in enumerate(self.stacks):
            if stack_id in overflow_types:
                # Overflow 색상의 스택은 자기 색상만 포함해야 함
                if not all(item == stack_id for item in stack):
                    return False
            else:
                # 나머지 스택은 아래에서부터 자기 색상, 그 위에 overflow 색상들만
                idx = 0
                # 자기 색상이 먼저 깔려있는지 확인
                while idx < len(stack) and stack[idx] == stack_id:
                    idx += 1
                # 그 위에 overflow 색상들만 있어야 함 (순서는 상관없음)
                while idx < len(stack) and stack[idx] in overflow_types:
                    idx += 1
                # 다른 색상이 있으면 실패
                if idx != len(stack):
                    return False
        return True

    def visualize(self) -> None:
        col_width = 7

        def fmt(text: str) -> str:
            return f"{text:^{col_width}}"

        print("".join(fmt("[TOP]") for _ in range(self.num_stacks)))
        for level in range(self.stack_capacity - 1, -1, -1):
            row = []
            for stack in self.stacks:
                cell = stack[level] if level < len(stack) else " "
                row.append(fmt(str(cell)))
            print("".join(row))
        print("".join(fmt("=====") for _ in range(self.num_stacks)))
        print("".join(fmt(f"S{i}") for i in range(self.num_stacks)))
        
if __name__ == "__main__":
    env = StackRearrangementEnv()
    env.visualize()