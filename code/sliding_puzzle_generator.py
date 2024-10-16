import random
from typing import List, Tuple, Optional

class SlidingPuzzleState:
    def __init__(self, size: int, board: Optional[List[List[int]]] = None):
        self.size = size
        if board:
            self.board = board
        else:
            self.board = self._generate_random_board()

    def _generate_random_board(self) -> List[List[int]]:
        numbers = list(range(1, self.size ** 2)) + [0]  # 0 represents the empty space
        random.shuffle(numbers)
        return [numbers[i:i+self.size] for i in range(0, len(numbers), self.size)]

    def get_empty_position(self) -> Tuple[int, int]:
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    return i, j
        raise ValueError("No empty position found")

    def get_possible_actions(self) -> List[str]:
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

    def apply_action(self, action: str) -> 'SlidingPuzzleState':
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

def generate_sliding_puzzle_problem(size: int, num_moves: int) -> Tuple[SlidingPuzzleState, SlidingPuzzleState]:
    goal_state = SlidingPuzzleState(size, [[(i * size + j + 1) % (size ** 2) for j in range(size)] for i in range(size)])
    initial_state = goal_state
    previous_action = None

    for _ in range(num_moves):
        actions = initial_state.get_possible_actions()
        if previous_action:
            actions.remove({"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}[previous_action]) # Prevent moving back
        action = random.choice(actions)
        initial_state = initial_state.apply_action(action)

    return initial_state, goal_state

def main():
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