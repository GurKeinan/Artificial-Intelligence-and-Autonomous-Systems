from sliding_puzzle_generator import SlidingPuzzleState
from typing import List, Tuple, Set

def manhattan_distance(state: SlidingPuzzleState, goal: SlidingPuzzleState) -> int:
    """
    Calculate the Manhattan distance heuristic.
    Sum of the distances each tile is from its goal position.
    """
    distance = 0
    for i in range(state.size):
        for j in range(state.size):
            tile = state.board[i][j]
            if tile != 0:  # Ignore the empty tile
                goal_i, goal_j = None, None
                for x in range(goal.size):
                    for y in range(goal.size):
                        if goal.board[x][y] == tile:
                            goal_i, goal_j = x, y
                            break
                    if goal_i is not None:
                        break
                distance += abs(i - goal_i) + abs(j - goal_j)
    return distance

def misplaced_tiles(state: SlidingPuzzleState, goal: SlidingPuzzleState) -> int:
    """
    Calculate the number of misplaced tiles heuristic.
    Count the number of tiles that are not in their goal position.
    """
    return sum(1 for i in range(state.size) for j in range(state.size)
               if state.board[i][j] != 0 and state.board[i][j] != goal.board[i][j])

class SlidingPuzzlePlanningProblem:
    def __init__(self, initial_state: SlidingPuzzleState, goal_state: SlidingPuzzleState):
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.size = initial_state.size

    def get_propositions(self, state: SlidingPuzzleState) -> Set[Tuple[int, int, int]]:
        """Return a set of propositions representing the current state."""
        props = set()
        for i in range(self.size):
            for j in range(self.size):
                tile = state.board[i][j]
                props.add((tile, i, j))
        return props

    def get_goal_propositions(self) -> Set[Tuple[int, int, int]]:
        """Return a set of propositions representing the goal state."""
        return self.get_propositions(self.goal_state)

def build_relaxed_planning_graph(problem: SlidingPuzzlePlanningProblem) -> List[Set[Tuple[int, int, int]]]:
    """Build a relaxed planning graph for the given problem."""
    proposition_layers = [problem.get_propositions(problem.initial_state)]
    goal_props = problem.get_goal_propositions()

    while True:
        current_props = proposition_layers[-1]
        new_props = set(current_props)

        for zero_tile, x, y in current_props:
            if zero_tile == 0:
                for tile, i, j in current_props:
                    # if i,j is near zero_i, zero_j
                    if (i == x and abs(j - y) == 1) or (j == y and abs(i - x) == 1):
                        new_props.add((tile, x, y))
                        new_props.add((0, i, j))

        if new_props == current_props:
            break  # No new propositions

        proposition_layers.append(new_props)

        if goal_props.issubset(new_props):
            break  # Goal achieved

    return proposition_layers

def h_max(state: SlidingPuzzleState, goal: SlidingPuzzleState) -> float:
    """
    Calculate the h_max heuristic.
    Returns the index of the first layer where all goal propositions are present.
    """
    problem = SlidingPuzzlePlanningProblem(state, goal)
    planning_graph = build_relaxed_planning_graph(problem)
    goal_props = problem.get_goal_propositions()

    for i, layer in enumerate(planning_graph):
        if goal_props.issubset(layer):
            return i  # Return the layer at which all goal propositions are achieved

    return float('inf')  # Goal not reachable

def h_ff(state: SlidingPuzzleState, goal: SlidingPuzzleState) -> float:
    """
    Calculate the FF (Fast Forward) heuristic.
    Counts the number of actions needed to achieve all goal propositions in the relaxed plan.
    """
    problem = SlidingPuzzlePlanningProblem(state, goal)
    planning_graph = build_relaxed_planning_graph(problem)
    goal_props = problem.get_goal_propositions()

    if not goal_props.issubset(planning_graph[-1]):
        return float('inf')  # Goal not reachable

    # Extract a relaxed plan (ignore mutex and delete effects)
    action_count = 0
    achieved_props = set()
    for layer in planning_graph:
        newly_achieved = goal_props - achieved_props
        newly_achieved = newly_achieved & layer
        action_count += len(newly_achieved)
        achieved_props.update(newly_achieved)
        if achieved_props == goal_props:
            break

    return action_count
