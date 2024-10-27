import random
from typing import List, Tuple, Optional

from general_state import StateInterface


class BlockWorldState(StateInterface):
    def __init__(self, num_blocks: int, num_stacks: int, stacks: Optional[List[List[int]]] = None):
        self.num_blocks = num_blocks
        self.num_stacks = num_stacks
        if stacks:
            self.stacks = stacks
        else:
            self.stacks = self._generate_random_stacks()

    def _generate_random_stacks(self) -> List[List[int]]:
        blocks = list(range(1, self.num_blocks + 1))
        random.shuffle(blocks)
        stacks = [[] for _ in range(self.num_stacks)]
        for block in blocks:
            stack_index = random.randint(0, self.num_stacks - 1)
            stacks[stack_index].append(block)
        return [stack for stack in stacks]  # Remove empty stacks

    def get_possible_actions(self) -> List[Tuple[int, int]]:
        actions = []
        for i, stack in enumerate(self.stacks):
            if stack:
                for j in range(len(self.stacks)):
                    if i != j:
                        actions.append((i, j))  # Move top block from stack i to stack j
        return actions

    def apply_action(self, action: Tuple[int, int]) -> 'BlockWorldState':
        from_stack, to_stack = action
        new_stacks = [stack[:] for stack in self.stacks]
        block = new_stacks[from_stack].pop()
        new_stacks[to_stack].append(block)
        return BlockWorldState(self.num_blocks, self.num_stacks, new_stacks)

    def __str__(self) -> str:
        return "|-  " + "\n|-  ".join(" ".join(str(block) for block in stack) for stack in self.stacks)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BlockWorldState):
            return NotImplemented
        return self.stacks == other.stacks

    def __hash__(self) -> int:
        return hash(tuple(tuple(stack) for stack in self.stacks))

def generate_block_world_problem(num_blocks: int, num_stacks: int, num_moves: int) -> Tuple[BlockWorldState, BlockWorldState]:
    goal_state = BlockWorldState(num_blocks, num_stacks)
    initial_state = goal_state
    visited_states = set()

    for _ in range(num_moves):
        actions = initial_state.get_possible_actions()
        action = random.choice(actions)
        new_state = initial_state.apply_action(action)

        num_of_tries = 0
        while new_state in visited_states and num_of_tries < 100:
            action = random.choice(actions)
            new_state = initial_state.apply_action(action)
            num_of_tries += 1

        visited_states.add(new_state)
        initial_state = new_state

    return initial_state, goal_state

def main():
    num_blocks = 5
    num_stacks = 3
    num_moves = 10
    initial_state, goal_state = generate_block_world_problem(num_blocks, num_stacks, num_moves)

    print("Initial State:")
    print(initial_state)
    print("\nGoal State:")
    print(goal_state)
    print("\nPossible actions from the initial state:")
    print(initial_state.get_possible_actions())

if __name__ == "__main__":
    main()