import random
from typing import List, Tuple, Optional

from general_state import StateInterface


class SlidingPuzzleState(StateInterface):
    """
    This class represents the state of a sliding puzzle problem. It contains the size of the puzzle and the board configuration.

    Attributes:
        size (int): The size of the puzzle.
        board (List[List[int]]): The board configuration.
    """
    def __init__(self, size, board):
        self.size = size
        if board:
            self.board = board
        else:
            self.board = self._generate_random_board()

    def _generate_random_board(self):
        """
        Generates a random board configuration for the sliding puzzle. The board configuration is a list of lists, where each inner list represents a row of the board. The numbers are shuffled and placed on the board, with 0 representing the empty space.
        """
        numbers = list(range(1, self.size ** 2)) + [0]  # 0 represents the empty space
        random.shuffle(numbers)
        return [numbers[i:i+self.size] for i in range(0, len(numbers), self.size)]

    def get_empty_position(self):
        """ Returns the row and column of the empty position on the board """
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    return i, j
        raise ValueError("No empty position found")

    def get_possible_actions(self):
        """
        Returns a list of possible actions that can be taken from the current state.

        The possible actions are "UP", "DOWN", "LEFT", and "RIGHT", depending on the position of the empty space.

        Returns:
            List[str]: A list of possible actions.
        """
        actions = []
        empty_row, empty_col = self.get_empty_position()

        if empty_row > 0:
            actions.append("UP")
        if empty_row < self.size - 1:
            actions.append("DOWN")
        if empty_col > 0:
            actions.append("LEFT")
        if empty_col < self.size - 1:
            actions.append("RIGHT")

        return actions

    def apply_action(self, action):
        """
        Applies the given action to the current state and returns a new state.

        Args:
            action (str): The action to apply. Must be one of "UP", "DOWN", "LEFT", or "RIGHT".

        Raises:
            ValueError: If the action is invalid.

        Returns:
            SlidingPuzzleState: The new state after applying the action.
        """
        empty_row, empty_col = self.get_empty_position()
        new_board = [row[:] for row in self.board]

        if action == "UP" and empty_row > 0:
            new_board[empty_row][empty_col], new_board[empty_row-1][empty_col] = new_board[empty_row-1][empty_col], new_board[empty_row][empty_col]
        elif action == "DOWN" and empty_row < self.size - 1:
            new_board[empty_row][empty_col], new_board[empty_row+1][empty_col] = new_board[empty_row+1][empty_col], new_board[empty_row][empty_col]
        elif action == "LEFT" and empty_col > 0:
            new_board[empty_row][empty_col], new_board[empty_row][empty_col-1] = new_board[empty_row][empty_col-1], new_board[empty_row][empty_col]
        elif action == "RIGHT" and empty_col < self.size - 1:
            new_board[empty_row][empty_col], new_board[empty_row][empty_col+1] = new_board[empty_row][empty_col+1], new_board[empty_row][empty_col]
        else:
            raise ValueError(f"Invalid action: {action}")

        return SlidingPuzzleState(self.size, new_board)

    def __str__(self) -> str:
        return "\n".join(" ".join(f"{num:2}" for num in row) for row in self.board)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SlidingPuzzleState):
            return NotImplemented
        return self.board == other.board

    def __hash__(self) -> int:
        return hash(tuple(tuple(row) for row in self.board))

def generate_sliding_puzzle_problem(size, num_moves):
    """
    Generates a sliding puzzle problem instance with the given size and number of moves.

    Args:
        size (int): The size of the puzzle (e.g., 3 for a 3x3 puzzle).
        num_moves (int): The number of random moves to make from the goal state to generate the initial state.

    Returns:
        Tuple[SlidingPuzzleState, SlidingPuzzleState]: A tuple containing the initial state and the goal state.
    """

    goal_state = SlidingPuzzleState(size, [[(i * size + j + 1) % (size ** 2) for j in range(size)] for i in range(size)])
    initial_state = goal_state
    visited_states = set()

    for _ in range(num_moves):
        actions = initial_state.get_possible_actions()
        action = random.choice(actions)
        new_state = initial_state.apply_action(action)

        while new_state in visited_states:
            action = random.choice(actions)
            new_state = initial_state.apply_action(action)

        visited_states.add(new_state)
        initial_state = new_state

    return initial_state, goal_state

def main():
    """ usage example of the sliding puzzle generator """
    size = 3
    num_moves = 20
    initial_state, goal_state = generate_sliding_puzzle_problem(size, num_moves)

    print("Initial State:")
    print(initial_state)
    print("\nGoal State:")
    print(goal_state)
    print("\nPossible actions from the initial state:")
    print(initial_state.get_possible_actions())

if __name__ == "__main__":
    main()