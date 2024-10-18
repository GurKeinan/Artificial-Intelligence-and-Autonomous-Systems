from block_world_generator import BlockWorldState
from typing import List, Tuple, Set, Dict
from collections import namedtuple


def misplaced_blocks(state: BlockWorldState, goal: BlockWorldState) -> int:
    """
    Calculate the number of misplaced blocks heuristic.
    Count the number of blocks that are not in their goal position.
    """
    misplaced = 0
    for i, stack in enumerate(state.stacks):
        for j, block in enumerate(stack):
            in_place_flag = False
            try:
                in_place_flag = block == goal.stacks[i][j]
            except IndexError:
                pass
            if not in_place_flag:
                misplaced += 1
    return misplaced


def height_difference(state: BlockWorldState, goal: BlockWorldState) -> int:
    """
    Calculate the total height difference heuristic.
    Sum of the differences between each block's current height and its goal height.
    Divide by 2 to account for double counting and make the heuristic admissible.
    """
    difference = 0
    for i, stack in enumerate(state.stacks):
        state_stack_height = len(stack)
        goal_stack_height = len(goal.stacks[i])
        difference += abs(state_stack_height - goal_stack_height)
    return difference // 2


Proposition = namedtuple('Proposition', ['block', 'position', 'top'])
Action = namedtuple('Action', ['block', 'from_pos', 'to_pos'])


class BlockWorldPlanningProblem:
    def __init__(self, initial_state: BlockWorldState, goal_state: BlockWorldState):
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.num_blocks = initial_state.num_blocks
        self.num_stacks = initial_state.num_stacks

    def get_propositions(self, state: BlockWorldState) -> Set[Proposition]:
        """Return a set of propositions representing the current state."""
        props = set()
        for i, stack in enumerate(state.stacks):
            for j, block in enumerate(stack):
                props.add(Proposition(block, (i, j), j == len(stack) - 1))
        return props

    def get_goal_propositions(self) -> Set[Proposition]:
        """Return a set of propositions representing the goal state."""
        return self.get_propositions(self.goal_state)


def build_relaxed_planning_graph(problem: BlockWorldPlanningProblem) -> Tuple[
    List[Set[Proposition]], List[Set[Action]]]:
    proposition_layers = [problem.get_propositions(problem.initial_state)]
    action_layers: List[Set[Action]] = []
    goal_props = problem.get_goal_propositions()

    while True:
        new_actions = set()
        props = proposition_layers[-1].copy()
        new_props = props.copy()

        for prop in props:
            if prop.top:
                for to_stack in range(problem.num_stacks):
                    if to_stack != prop.position[0]:
                        new_action = Action(prop.block, prop.position, (to_stack, 0))
                        new_actions.add(new_action)
                        new_props.add(Proposition(prop.block, (to_stack, 0), True))
                        # Update the 'top' status of the block underneath
                        if prop.position[1] > 0:
                            underneath_block = next(
                                (p for p in props if p.position == (prop.position[0], prop.position[1] - 1)), None)
                            if underneath_block:
                                new_props.add(Proposition(underneath_block.block, underneath_block.position, True))

        action_layers.append(new_actions)
        proposition_layers.append(new_props)

        if goal_props.issubset(new_props):
            break

    return proposition_layers, action_layers


def h_max(state: BlockWorldState, goal: BlockWorldState) -> int:
    """Return the h_max heuristic value for the given problem."""
    problem = BlockWorldPlanningProblem(state, goal)
    prop_layers, act_layers = build_relaxed_planning_graph(problem)
    goal_props = problem.get_goal_propositions()

    prop_layers_costs = [{prop: 0 for prop in prop_layers[0]}]
    action_layers_costs = []

    for index, (prop_layer, act_layer) in enumerate(zip(prop_layers[1:], act_layers)):
        action_layer_cost = {}
        for action in act_layer:
            needed_props = [prop for prop in prop_layers[index] if
                            prop.block == action.block and prop.position == action.from_pos and prop.top]
            action_cost = max(prop_layers_costs[-1][prop] for prop in needed_props) + 1
            action_layer_cost[action] = action_cost
        action_layers_costs.append(action_layer_cost)

        prop_layer_cost = {}
        for prop in prop_layer:
            if prop in prop_layers_costs[index]:
                prop_layer_cost[prop] = prop_layers_costs[index][prop]
                continue
            possible_actions = [action for action in act_layer if
                                action.block == prop.block and action.to_pos == prop.position]
            prop_cost = min(action_layer_cost[action] for action in possible_actions) if possible_actions else float(
                'inf')
            prop_layer_cost[prop] = prop_cost
        prop_layers_costs.append(prop_layer_cost)

    return max(prop_layers_costs[-1][prop] for prop in goal_props)
