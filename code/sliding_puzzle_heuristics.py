from sliding_puzzle_generator import SlidingPuzzleState
from typing import List, Tuple, Set, Dict
from collections import namedtuple


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


Proposition = namedtuple('Proposition', ['tile', 'location_x', 'location_y'])
Action = namedtuple('Action', ['tile', 'before_x', 'before_y', 'after_x', 'after_y'])


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
                props.add(Proposition(tile, i, j))
        return props

    def get_goal_propositions(self) -> Set[Tuple[int, int, int]]:
        """Return a set of propositions representing the goal state."""
        return self.get_propositions(self.goal_state)


def build_relaxed_planning_graph(problem: SlidingPuzzlePlanningProblem) -> Tuple[List[Set[Tuple[int, int, int]]], List[Set[str]]]:
    """Build a relaxed planning graph for the given problem."""
    proposition_layers = [problem.get_propositions(problem.initial_state)]
    action_layers: List[Set[str]] = []
    goal_props = problem.get_goal_propositions()

    while True:
        if len(action_layers) > 0:
            new_actions = action_layers[-1].copy()
        else:
            new_actions = set()

        props = proposition_layers[-1].copy()
        new_props = props.copy()

        for zero_tile, x, y in props:
            if zero_tile == 0:
                for tile, i, j in props:
                    if (i == x and abs(j - y) == 1) or (j == y and abs(i - x) == 1):
                        new_actions.add(Action(tile, i, j, x, y))
                        new_props.add(Proposition(tile, x, y))
                        new_props.add(Proposition(0, i, j))

        action_layers.append(new_actions)
        proposition_layers.append(new_props)

        if goal_props.issubset(new_props):
            break

    return proposition_layers, action_layers


def h_max(state: SlidingPuzzleState, goal: SlidingPuzzleState) -> int:
    """Return the h_max heuristic value for the given problem."""
    problem = SlidingPuzzlePlanningProblem(state, goal)
    prop_layers, act_layers = build_relaxed_planning_graph(problem)
    goal_props = problem.get_goal_propositions()

    prop_layers_costs = [{prop : 0 for prop in prop_layers[0]}]
    action_layers_costs = []

    for index, (prop_layer, act_layer) in enumerate(zip(prop_layers[1:], act_layers)):
        action_layer_cost = {}
        for action in act_layer:
            # the needed proposition are 0 in the after location and tile in the before location
            needed_props = [Proposition(action.tile, action.before_x, action.before_y),
                            Proposition(0, action.after_x, action.after_y)]
            action_cost = max(prop_layers_costs[-1][prop] for prop in needed_props) + 1
            action_layer_cost[action] = action_cost
        action_layers_costs.append(action_layer_cost)

        prop_layer_cost = {}
        for prop in prop_layer:
            # if it is in the previous layer, the cost is the same as the previous layer
            if prop in prop_layers_costs[index]:
                prop_layer_cost[prop] = prop_layers_costs[index][prop]
                continue
            # the actions that can achieve the proposition
            if prop.tile == 0:
                possible_actions = [action for action in act_layer if (action.before_x == prop.location_x and action.before_y == prop.location_y)]
            else:
                possible_actions = [action for action in act_layer if (action.tile == prop.tile and action.after_x == prop.location_x and action.after_y == prop.location_y)]
            prop_cost = min(action_layer_cost[action] for action in possible_actions)
            prop_layer_cost[prop] = prop_cost
        prop_layers_costs.append(prop_layer_cost)

    return max(prop_layers_costs[-1][prop] for prop in goal_props)








