from collections import Counter
from typing import List, Optional, Sequence, Union


class StackRearrangementEnv:
    """
    - num_stacks: automatically determined by len(stack_capacities)
    - stack_capacities[i]: maximum number of elements in stack i
    - stacks[i]: elements of stack i (bottom->top order, pop() is from the top)
    """

    def __init__(
        self,
        stacks: Optional[List[List[int]]] = None,
        stack_capacities: Optional[Sequence[int]] = None,
        # Backward compatibility: allow creation with num_stacks/stack_capacity as before
        num_stacks: Optional[int] = None,
        stack_capacity: Optional[Union[int, Sequence[int]]] = None,
    ):
        # 1) Determine cap
        if stack_capacities is None:
            if stack_capacity is None:
                raise ValueError("stack_capacities or stack_capacity must be provided.")

            # stack_capacity is int, uniform cap; if list, per-stack cap
            if isinstance(stack_capacity, int):
                if num_stacks is None:
                    if stacks is None:
                        raise ValueError("num_stacks or stacks is required.")
                    num_stacks = len(stacks)
                stack_capacities = [int(stack_capacity)] * int(num_stacks)
            else:
                stack_capacities = [int(x) for x in stack_capacity]

        self.stack_capacities: List[int] = [int(x) for x in stack_capacities]
        if any(c < 0 for c in self.stack_capacities):
            raise ValueError(f"stack_capacities must be 0 or greater: {self.stack_capacities}")

        self.num_stacks = len(self.stack_capacities)

        # 2) Determine stacks
        if stacks is None:
            self.stacks = [[] for _ in range(self.num_stacks)]
        else:
            if len(stacks) != self.num_stacks:
                raise ValueError(
                    f"stacks length ({len(stacks)}) and stack_capacities length ({self.num_stacks}) are different."
                )
            self.stacks = [list(s) for s in stacks]

        # 3) Validity check
        self._validate_lengths()

    def _validate_lengths(self) -> None:
        for i, s in enumerate(self.stacks):
            if len(s) > self.stack_capacities[i]:
                raise ValueError(
                    f"Stack {i} length ({len(s)}) exceeded cap ({self.stack_capacities[i]})."
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
        # Defense against invalid states entered from outside
        for i, s in enumerate(self.stacks):
            if len(s) > self.cap(i):
                return False

        # 1) Basic condition: each stack contains only its own index color
        if all(all(item == stack_id for item in stack) for stack_id, stack in enumerate(self.stacks)):
            return True

        # 2) overflow judgment: "total count of color i > stack i cap"
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
            # Exceeding cap is an impossible state to begin with
            if len(stack) > self.cap(stack_id):
                return False

            if stack_id in overflow_types:
                # overflow stack contains only its own color
                if not all(item == stack_id for item in stack):
                    return False
            else:
                # bottom: its own color, top: only overflow colors
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

        # Print based on max height, but treat stack areas with smaller cap as blank
        for level in range(h - 1, -1, -1):
            row = []
            for sid, stack in enumerate(self.stacks):
                if level >= self.cap(sid):
                    cell = " "  # This stack does not have this height itself
                else:
                    cell = stack[level] if level < len(stack) else " "
                row.append(fmt(str(cell)))
            print("".join(row))

        print("".join(fmt("=====") for _ in range(self.num_stacks)))
        print("".join(fmt(f"S{i}({self.cap(i)})") for i in range(self.num_stacks)))