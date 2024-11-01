from typing import Set, Tuple, List, Union
from collections import namedtuple

from sympy import O
from block_world_generator import BlockWorldState


def bw_misplaced_blocks(state, goal):
    """Calculate the number of misplaced blocks heuristic.
    A block is misplaced if it's not in the same position as in the goal state.

    Args:
        state (BlockWorldState): Current state
        goal (BlockWorldState): Goal state

    Returns:
        int: Number of misplaced blocks
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


def bw_height_difference(state, goal):
    """Calculate the total height difference heuristic.
    Sum the absolute difference between stack heights divided by 2.

    Args:
        state (BlockWorldState): Current state
        goal (BlockWorldState): Goal state

    Returns:
        int: Total height difference heuristic value
    """
    difference = 0
    for i, stack in enumerate(state.stacks):
        state_stack_height = len(stack)
        goal_stack_height = len(goal.stacks[i])
        difference += abs(state_stack_height - goal_stack_height)
    return difference // 2


OnProp = namedtuple('OnProp', ['above_block', 'below_block'])
OnTableProp = namedtuple('OnTableProp', ['block'])
ClearProp = namedtuple('ClearProp', ['block'])
EmptyProp = namedtuple('EmptyProp', ['stack'])
InStackProp = namedtuple('InStackProp', ['block', 'stack'])
MoveAction = namedtuple('MoveAction', [
                        'block', 'below_block_old', 'from_stack', 'to_stack', 'below_block_new'])


def get_propositions(state):
    """Get all propositions from a given BlockWorldState.

    Args:
        state (BlockWorldState): Current block world state

    Returns:
        Tuple[Set]: Returns sets of propositions (on_props, on_table_props, clear_props, empty_props, in_stack_props)
    """
    on_props = set()
    on_table_props = set()
    clear_props = set()
    empty_props = set()
    in_stack_props = set()

    for stack_idx, stack in enumerate(state.stacks):
        if len(stack) == 0:
            empty_props.add(EmptyProp(stack_idx))
        for block_idx, block in enumerate(stack):
            in_stack_props.add(InStackProp(block, stack_idx))
            if block_idx == 0:
                on_table_props.add(OnTableProp(block))
            if block_idx == len(stack) - 1:
                clear_props.add(ClearProp(block))
            if block_idx > 0:
                on_props.add(OnProp(block, stack[block_idx - 1]))

    return on_props, on_table_props, clear_props, empty_props, in_stack_props


def check_move_action_validity(action, propositions_tuple):
    """Check if a move action is valid given the current propositions.

    Args:
        action (MoveAction): Action to check (block, below_block_old, from_stack, to_stack, below_block_new)
        propositions_tuple (Tuple[Set]): Current propositions (on_props, on_table_props, clear_props, empty_props, in_stack_props)

    Returns:
        bool: True if the action is valid, False otherwise
    """
    on_props, on_table_props, clear_props, empty_props, in_stack_props = propositions_tuple
    block, below_block_old, from_stack, to_stack, below_block_new = action

    if InStackProp(block, from_stack) not in in_stack_props:  # block is not in the from_stack
        return False
    # below_block_old is not in the from_stack
    if InStackProp(below_block_old, from_stack) not in in_stack_props and below_block_old is not None:
        return False
    # below_block_new is not in the to_stack
    if InStackProp(below_block_new, to_stack) not in in_stack_props and below_block_new is not None:
        return False
    if ClearProp(block) not in clear_props:  # block is not clear
        return False
    # below_block_new is not clear
    if ClearProp(below_block_new) not in clear_props and below_block_new is not None:
        return False
    # block is not on below_block_old
    if below_block_old is not None and OnProp(block, below_block_old,) not in on_props:
        return False
    # block don't have below_block_old and is not on table
    if below_block_old is None and not OnTableProp(block) in on_table_props:
        return False
    # block don't have below_block_new and to_stack is not empty
    if below_block_new is None and not EmptyProp(to_stack) in empty_props:
        return False
    if from_stack == to_stack:
        return False

    return True


def add_propositions(action, propositions_tuple) :
    """Add propositions that result from applying the given action.

    Args:
        action (MoveAction): The action being applied
        propositions_tuple (Tuple[Set]): Current proposition sets (on_props, on_table_props, clear_props, empty_props, in_stack_props)

    Returns:
        Tuple[Set]: Updated proposition sets after applying the action
    """
    on_props, on_table_props, clear_props, empty_props, in_stack_props = propositions_tuple
    new_on_props = on_props.copy()
    new_on_table_props = on_table_props.copy()
    new_clear_props = clear_props.copy()
    new_empty_props = empty_props.copy()
    new_in_stack_props = in_stack_props.copy()

    block, below_block_old, from_stack, to_stack, below_block_new = action

    new_in_stack_props.add(InStackProp(block, to_stack))
    if below_block_old is not None:
        new_clear_props.add(ClearProp(below_block_old))
    else:
        new_empty_props.add(EmptyProp(from_stack))

    if below_block_new is not None:
        new_on_props.add(OnProp(block, below_block_new))
    else:
        new_on_table_props.add(OnTableProp(block))

    return new_on_props, new_on_table_props, new_clear_props, new_empty_props, new_in_stack_props


def get_actions(propositions_tuple):
    """Get all valid actions for the given propositions.

    Args:
        propositions_tuple (Tuple[Set]): Current propositions (on_props, on_table_props, clear_props, empty_props, in_stack_props)

    Returns:
        Set[MoveAction]: Set of valid move actions
    """
    on_props, on_table_props, clear_props, empty_props, in_stack_props = propositions_tuple
    actions = set()

    # Get all blocks and stacks
    blocks = {prop.block for prop in in_stack_props}
    stacks = {prop.stack for prop in in_stack_props}.union(
        {prop.stack for prop in empty_props})

    for block in blocks:
        if ClearProp(block) in clear_props:
            from_stack = next(
                prop.stack for prop in in_stack_props if prop.block == block)

            # Determine the block below (if any)
            below_block_old = next(
                (prop.below_block for prop in on_props if prop.above_block == block), None)

            for to_stack in stacks:
                if to_stack != from_stack:
                    # Move to an empty stack
                    if EmptyProp(to_stack) in empty_props:
                        action = MoveAction(
                            block, below_block_old, from_stack, to_stack, None)
                        if check_move_action_validity(action, propositions_tuple):
                            actions.add(action)

                    # Move on top of another block
                    for below_block_new in blocks:
                        if ClearProp(below_block_new) in clear_props and below_block_new != block:
                            if InStackProp(below_block_new, to_stack) in in_stack_props:
                                action = MoveAction(
                                    block, below_block_old, from_stack, to_stack, below_block_new)
                                if check_move_action_validity(action, propositions_tuple):
                                    actions.add(action)

    return actions


class BlockWorldPlanningProblem:
    """ A class representing a Block World planning problem.

    Attributes:
        initial_state (BlockWorldState): The initial state of the problem
        goal_state (BlockWorldState): The goal state of the problem
        num_blocks (int): The number of blocks in the problem
        num_stacks (int): The number of stacks in the problem
    """

    def __init__(self, initial_state: BlockWorldState, goal_state: BlockWorldState):
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.num_blocks = initial_state.num_blocks
        self.num_stacks = initial_state.num_stacks

    def get_initial_propositions(self) -> Tuple[Set[OnProp],
                                                      Set[OnTableProp],
                                                        Set[ClearProp],
                                                        Set[EmptyProp],
                                                        Set[InStackProp]]:
        """Return a set of propositions representing the initial state."""
        return get_propositions(self.initial_state)

    def get_goal_propositions(self):
        """Return a set of propositions representing the goal state."""
        return get_propositions(self.goal_state)

    def check_goal_propositions(self, propositions):
        """Check if the current propositions satisfy the goal propositions.

        Args:
            propositions (Tuple[Set]): Current propositions (on_props, on_table_props, clear_props, empty_props, in_stack_props)

        Returns:
            bool: True if goal propositions are satisfied, False otherwise
        """
        goal_props = self.get_goal_propositions()
        goal_on_props, goal_on_table_props, goal_clear_props, goal_empty_props, goal_in_stack_props = goal_props
        on_props, on_table_props, clear_props, empty_props, in_stack_props = propositions

        return (goal_on_props.issubset(on_props)
                and goal_on_table_props.issubset(on_table_props)
                and goal_clear_props.issubset(clear_props)
                and goal_empty_props.issubset(empty_props)
                and goal_in_stack_props.issubset(in_stack_props))


def build_relaxed_planning_graph(problem):
    """Build a relaxed planning graph for the Block World problem.
    A relaxed planning graph ignores delete effects of actions.

    Args:
        problem (BlockWorldPlanningProblem): The Block World planning problem

    Returns:
        Tuple[List[Set], List[Set]]: A tuple containing:
            - List of proposition layers
            - List of action layers
    """
    propositions = problem.get_initial_propositions()
    proposition_layers = [propositions]
    action_layers = []
    goal_props = problem.get_goal_propositions()

    while True:
        actions = get_actions(proposition_layers[-1])
        for action in actions:
            propositions = add_propositions(action, propositions)
        action_layers.append(actions)
        proposition_layers.append(propositions)

        if problem.check_goal_propositions(propositions):
            break

    return proposition_layers, action_layers


def bw_h_max(block_world_state, block_world_goal_state):
    """Calculate the h_max heuristic value for a Block World state.
    This heuristic estimates the minimum number of steps needed to reach the goal
    using a relaxed planning graph where delete effects are ignored.

    Args:
        block_world_state (BlockWorldState): Current state of the block world
        block_world_goal_state (BlockWorldState): Goal state of the block world

    Returns:
        int: The h_max heuristic value
    """
    problem = BlockWorldPlanningProblem(
        block_world_state, block_world_goal_state)
    goal_props = problem.get_goal_propositions()
    prop_layers, action_layers = build_relaxed_planning_graph(problem)

    prop_costs = [{prop: 0 for prop in set.union(*prop_layers[0])}]
    action_costs = []

    for action_layer, prop_layer in zip(action_layers, prop_layers[1:]):
        action_costs_dict = dict()
        for action in action_layer:
            preconditions = get_action_preconditions(action)
            action_costs_dict[action] = max(
                prop_costs[-1][p] for p in preconditions) + 1
        action_costs.append(action_costs_dict)

        prop_costs_dict = dict()
        for prop in set.union(*prop_layer):
            if prop in prop_costs[-1]:
                prop_costs_dict[prop] = prop_costs[-1][prop]
            else:
                action_achievers = get_prop_achievers(prop, action_layer)
                prop_costs_dict[prop] = min(
                    action_costs_dict[action] for action in action_achievers)

        prop_costs.append(prop_costs_dict)

    return max(prop_costs[-1][prop] for prop in set.union(*goal_props))


def get_action_preconditions(action):
    """Return the set of preconditions for the given action.

    Args:
        action (MoveAction): The action to get preconditions for

    Returns:
        Set[Union[OnProp, OnTableProp, ClearProp, EmptyProp, InStackProp]]: Set of propositions that must be true for the action to be valid
    """
    block, below_block_old, from_stack, to_stack, below_block_new = action

    preconditions = {
        InStackProp(block, from_stack),
        ClearProp(block)
    }

    if below_block_old is not None:
        preconditions.add(OnProp(block, below_block_old))
    else:
        preconditions.add(OnTableProp(block))

    if below_block_new is not None:
        preconditions.add(ClearProp(below_block_new))
        preconditions.add(InStackProp(below_block_new, to_stack))
    else:
        preconditions.add(EmptyProp(to_stack))

    return preconditions


def get_prop_achievers(prop, actions):
    """Return actions that achieve the given proposition.

    Args:
        prop (Union[OnProp, OnTableProp, ClearProp, EmptyProp, InStackProp]): The proposition
        actions (Set[MoveAction]): Set of available actions

    Returns:
        Set[MoveAction]: Set of actions that achieve the proposition
    """
    if isinstance(prop, OnProp):
        return get_on_prop_achievers(prop, actions)
    elif isinstance(prop, OnTableProp):
        return get_on_table_prop_achievers(prop, actions)
    elif isinstance(prop, ClearProp):
        return get_clear_prop_achievers(prop, actions)
    elif isinstance(prop, EmptyProp):
        return get_empty_prop_achievers(prop, actions)
    elif isinstance(prop, InStackProp):
        return get_in_stack_prop_achievers(prop, actions)


def get_on_prop_achievers(on_prop, actions):
    """Return the set of actions that achieve the given OnProp proposition.

    Args:
        on_prop (OnProp): The OnProp proposition
        actions (Set[MoveAction]): Set of available actions

    Returns:
        Set[MoveAction]: Set of actions that achieve the proposition
    """
    return {action for action in actions if action.block == on_prop.above_block and action.below_block_new == on_prop.below_block}


def get_on_table_prop_achievers(on_table_prop, actions):
    """Return the set of actions that achieve the given OnTableProp proposition.

    Args:
        on_table_prop (OnTableProp): The OnTableProp proposition
        actions (Set[MoveAction]): Set of available actions

    Returns:
        Set[MoveAction]: Set of actions that achieve the proposition
    """
    return {action for action in actions if action.block == on_table_prop.block and action.below_block_new is None}


def get_clear_prop_achievers(clear_prop, actions):
    """Return the set of actions that achieve the given ClearProp proposition.

    Args:
        clear_prop (ClearProp): The ClearProp proposition
        actions (Set[MoveAction]): Set of available actions

    Returns:
        Set[MoveAction]: Set of actions that achieve the proposition
    """
    return {action for action in actions if action.below_block_old == clear_prop.block}


def get_empty_prop_achievers(empty_prop, actions):
    """Return the set of actions that achieve the given EmptyProp proposition.

    Args:
        empty_prop (EmptyProp): The EmptyProp proposition to check
        actions (Set[MoveAction]): Set of available actions

    Returns:
        Set[MoveAction]: Set of actions that achieve the empty stack proposition
    """
    return {action for action in actions if action.from_stack == empty_prop.stack and action.below_block_old is None}


def get_in_stack_prop_achievers(in_stack_prop, actions):
    """Return the set of actions that achieve the given InStackProp proposition.

    Args:
        in_stack_prop (InStackProp): The InStackProp proposition to check
        actions (Set[MoveAction]): Set of available actions

    Returns:
        Set[MoveAction]: Set of actions that achieve the in-stack proposition
    """
    return {action for action in actions if action.block == in_stack_prop.block and action.to_stack == in_stack_prop.stack}
