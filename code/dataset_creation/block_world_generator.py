"""
This module contains the implementation of a Block World problem generator.

Classes:
    BlockWorldState: The state of the Block World with a given number of blocks and stacks.

Functions:
    generate_block_world_problem(num_blocks: int, num_stacks: int,
    num_moves: int) -> Tuple[BlockWorldState, BlockWorldState]:
        Generates an initial and goal state for the Block World problem
        with a specified number of blocks, stacks, and moves.

    main():
        Demonstrates the generation of a Block World problem and prints the initial and goal states
        along with possible actions from the initial state.
"""

import random
from typing import List, Tuple, Optional
from general_state import StateInterface


class BlockWorldState(StateInterface):
    """
    Represents the state of a block world in which blocks can be stacked on top of each other.
    Attributes:
        num_blocks (int): The number of blocks in the block world.
        num_stacks (int): The number of stacks in the block world.
        stacks (List[List[int]]): The current configuration of blocks in stacks.
    Methods:
        __init__(num_blocks: int, num_stacks: int, stacks: Optional[List[List[int]]] = None):
            Initializes a new BlockWorldState with the given number of blocks and stacks.
            If stacks are not provided, generates a random configuration of blocks in stacks.
        _generate_random_stacks() -> List[List[int]]:
            Generates a random configuration of blocks in stacks.
        get_possible_actions() -> List[Tuple[int, int]]:
            Returns a list of possible actions that can be taken from the current state.
            Each action is represented as a tuple (i, j) where a block is
            moved from stack i to stack j.
        apply_action(action: Tuple[int, int]) -> 'BlockWorldState':
            Applies the given action to the current state and returns a new BlockWorldState.
        __str__() -> str:
            Returns a string representation of the current state.
        __eq__(other: object) -> bool:
            Checks if the current state is equal to another BlockWorldState.
        __hash__() -> int:
            Returns a hash value for the current state.
    """

    def __init__(self, num_blocks: int, num_stacks: int, stacks: Optional[List[List[int]]] = None):
        self.num_blocks = num_blocks
        self.num_stacks = num_stacks
        if stacks:
            self.stacks = stacks
        else:
            self.stacks = self._generate_random_stacks()

    def _generate_random_stacks(self) -> List[List[int]]:
        """
        Generates a random configuration of blocks stacked in multiple stacks.
        This method creates a list of stacks, each containing a random distribution
        of blocks. The blocks are shuffled and then distributed randomly across the
        stacks.
        Returns:
            List[List[int]]: A list of stacks, where each stack is a list of block
            numbers. Empty stacks are removed from the final output.
        """

        blocks = list(range(1, self.num_blocks + 1))
        random.shuffle(blocks)
        stacks = [[] for _ in range(self.num_stacks)]
        for block in blocks:
            stack_index = random.randint(0, self.num_stacks - 1)
            stacks[stack_index].append(block)
        return [stack for stack in stacks]  # Remove empty stacks

    def get_possible_actions(self) -> List[Tuple[int, int]]:
        """
        Generate a list of possible actions in the block world.
        Each action is represented as a tuple (i, j) where a block is moved from
        the top of stack i to the top of stack j.
        Returns:
            List[Tuple[int, int]]: A list of tuples representing possible actions.
        """

        actions = []
        for i, stack in enumerate(self.stacks):
            if stack:
                for j in range(len(self.stacks)):
                    if i != j:
                        actions.append((i, j))  # Move top block from stack i to stack j
        return actions

    def apply_action(self, action: Tuple[int, int]) -> 'BlockWorldState':
        """
        Applies the given action to the current block world state and returns a new state.
        Args:
            action (Tuple[int, int]): A tuple representing the action to be applied.
            The first element is the index of the stack to move a block from,
            and the second element is the index of the stack to move the block to.
        Returns:
            BlockWorldState: A new BlockWorldState object
            representing the state after the action is applied.
        """

        from_stack, to_stack = action
        new_stacks = [stack[:] for stack in self.stacks]
        block = new_stacks[from_stack].pop()
        new_stacks[to_stack].append(block)
        return BlockWorldState(self.num_blocks, self.num_stacks, new_stacks)

    def __str__(self) -> str:
        return "|-  " + "\n|-  ".join(" ".join(str(block) for block in stack)
                                      for stack in self.stacks)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BlockWorldState):
            return NotImplemented
        return self.stacks == other.stacks

    def __hash__(self) -> int:
        return hash(tuple(tuple(stack) for stack in self.stacks))

def generate_block_world_problem(num_blocks: int, num_stacks: int,
                                 num_moves: int) -> Tuple[BlockWorldState, BlockWorldState]:
    """
    Generates a block world problem with a specified number of blocks, stacks, and moves.
    Args:
        num_blocks (int): The number of blocks in the block world.
        num_stacks (int): The number of stacks in the block world.
        num_moves (int): The number of moves to perform to generate the initial state.
    Returns:
        Tuple[BlockWorldState, BlockWorldState]: A tuple containing the initial state
        and the goal state of the block world.
    """

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
    """
    Main function to demonstrate the generation of a block world problem.
    """

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
