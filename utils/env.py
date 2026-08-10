from collections import Counter
from typing import List, Optional, Sequence, Union


class StackRearrangementEnv:
    """
    - num_stacks: len(stack_capacities)로 자동 결정
    - stack_capacities[i]: i번 스택의 최대 원소 수
    - stacks[i]: i번 스택의 원소들(아래->위 순서, pop()은 맨 위)
    """

    def __init__(
        self,
        stacks: Optional[List[List[int]]] = None,
        stack_capacities: Optional[Sequence[int]] = None,
        # 하위호환: 예전처럼 num_stacks/stack_capacity로도 만들 수 있게
        num_stacks: Optional[int] = None,
        stack_capacity: Optional[Union[int, Sequence[int]]] = None,
    ):
        # 1) cap 결정
        if stack_capacities is None:
            if stack_capacity is None:
                raise ValueError("stack_capacities 또는 stack_capacity를 제공해야 합니다.")

            # stack_capacity가 int면 균일 cap, list면 per-stack cap
            if isinstance(stack_capacity, int):
                if num_stacks is None:
                    if stacks is None:
                        raise ValueError("num_stacks 또는 stacks가 필요합니다.")
                    num_stacks = len(stacks)
                stack_capacities = [int(stack_capacity)] * int(num_stacks)
            else:
                stack_capacities = [int(x) for x in stack_capacity]

        self.stack_capacities: List[int] = [int(x) for x in stack_capacities]
        if any(c < 0 for c in self.stack_capacities):
            raise ValueError(f"stack_capacities는 0 이상이어야 합니다: {self.stack_capacities}")

        self.num_stacks = len(self.stack_capacities)

        # 2) stacks 결정
        if stacks is None:
            self.stacks = [[] for _ in range(self.num_stacks)]
        else:
            if len(stacks) != self.num_stacks:
                raise ValueError(
                    f"stacks 길이({len(stacks)})와 stack_capacities 길이({self.num_stacks})가 다릅니다."
                )
            self.stacks = [list(s) for s in stacks]

        # 3) 유효성 체크
        self._validate_lengths()

    def _validate_lengths(self) -> None:
        for i, s in enumerate(self.stacks):
            if len(s) > self.stack_capacities[i]:
                raise ValueError(
                    f"스택 {i} 길이({len(s)})가 cap({self.stack_capacities[i]})을 초과했습니다."
                )

    def cap(self, stack_id: int) -> int:
        return self.stack_capacities[stack_id]

    def max_cap(self) -> int:
        return max(self.stack_capacities) if self.stack_capacities else 0

    def peek(self, stack_id: int) -> Optional[int]:
        s = self.stacks[stack_id]
        return s[-1] if s else None

    def pop(self, stack_id: int) -> Optional[int]:
        if not self.stacks[stack_id]:
            return None
        return self.stacks[stack_id].pop()

    def push(self, stack_id: int, item: int) -> bool:
        if len(self.stacks[stack_id]) >= self.cap(stack_id):
            return False
        self.stacks[stack_id].append(item)
        return True

    def move(self, src: int, dst: int) -> bool:
        if src == dst or not self.stacks[src]:
            return False
        if len(self.stacks[dst]) >= self.cap(dst):
            return False
        item = self.stacks[src].pop()
        self.stacks[dst].append(item)
        return True

    def is_goal(self, goal_state: List[List[int]]) -> bool:
        return self.stacks == goal_state

    def is_solved(self) -> bool:
        # 혹시 외부에서 잘못 넣은 상태 방어
        for i, s in enumerate(self.stacks):
            if len(s) > self.cap(i):
                return False

        # 1) 기본 조건: 각 스택이 자기 인덱스 색만 포함
        if all(all(item == stack_id for item in stack) for stack_id, stack in enumerate(self.stacks)):
            return True

        # 2) overflow 판정: "색 i의 전체 개수 > i번 스택 cap"
        counts = Counter(item for stack in self.stacks for item in stack)

        overflow_types = set()
        for item_type, total in counts.items():
            if 0 <= item_type < self.num_stacks and total > self.cap(item_type):
                overflow_types.add(item_type)

        if overflow_types:
            return self._is_valid_with_overflow(overflow_types)

        return False

    def _is_valid_with_overflow(self, overflow_types: set[int]) -> bool:
        for stack_id, stack in enumerate(self.stacks):
            # cap 초과는 애초에 불가능 상태
            if len(stack) > self.cap(stack_id):
                return False

            if stack_id in overflow_types:
                # overflow 스택은 자기 색상만
                if not all(item == stack_id for item in stack):
                    return False
            else:
                # 아래: 자기 색상, 위: overflow 색상들만
                idx = 0
                while idx < len(stack) and stack[idx] == stack_id:
                    idx += 1
                while idx < len(stack) and stack[idx] in overflow_types:
                    idx += 1
                if idx != len(stack):
                    return False
        return True

    def visualize(self) -> None:
        col_width = 7

        def fmt(text: str) -> str:
            return f"{text:^{col_width}}"

        h = self.max_cap()
        print("".join(fmt("[TOP]") for _ in range(self.num_stacks)))

        # max 높이 기준으로 찍되, cap이 더 작은 스택 영역은 공백 처리
        for level in range(h - 1, -1, -1):
            row = []
            for sid, stack in enumerate(self.stacks):
                if level >= self.cap(sid):
                    cell = " "  # 이 스택은 이 높이 자체가 없음
                else:
                    cell = stack[level] if level < len(stack) else " "
                row.append(fmt(str(cell)))
            print("".join(row))

        print("".join(fmt("=====") for _ in range(self.num_stacks)))
        print("".join(fmt(f"S{i}({self.cap(i)})") for i in range(self.num_stacks)))