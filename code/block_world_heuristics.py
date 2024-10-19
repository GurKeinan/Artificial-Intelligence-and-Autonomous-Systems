import copy

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



OnProp = namedtuple('OnProp', ['above_block', 'below_block'])
OnTableProp = namedtuple('OnTableProp', ['block'])
ClearProp = namedtuple('ClearProp', ['block'])
EmptyProp = namedtuple('EmptyProp', ['stack'])
InStackProp = namedtuple('InStackProp', ['block', 'stack'])
MoveAction = namedtuple('MoveAction', ['block', 'below_block_old', 'from_stack', 'to_stack', 'below_block_new'])

def get_propositions(state: BlockWorldState) -> Tuple[Set[namedtuple],...]:
    """Return a set of propositions representing the current state."""
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

def check_move_action_validity(action: MoveAction, propositions_tuple: Tuple[Set[namedtuple],...]) -> bool:
    """Return True if the given move action is valid in the current state."""
    on_props, on_table_props, clear_props, empty_props, in_stack_props = propositions_tuple
    block, below_block_old, from_stack, to_stack, below_block_new = action

    if InStackProp(block, from_stack) not in in_stack_props: # block is not in the from_stack
        return False
    if InStackProp(below_block_old, from_stack) not in in_stack_props and below_block_old is not None: # below_block_old is not in the from_stack
        return False
    if InStackProp(below_block_new, to_stack) not in in_stack_props and below_block_new is not None: # below_block_new is not in the to_stack
        return False
    if ClearProp(block) not in clear_props: # block is not clear
        return False
    if ClearProp(below_block_new) not in clear_props and below_block_new is not None: # below_block_new is not clear
        return False
    if below_block_old is not None and OnProp(block, below_block_old,) not in on_props: # block is not on below_block_old
        return False
    if below_block_old is None and not OnTableProp(block) in on_table_props: # block don't have below_block_old and is not on table
        return False
    if below_block_new is None and not EmptyProp(to_stack) in empty_props: # block don't have below_block_new and to_stack is not empty
        return False
    if from_stack == to_stack:
        return False

    return True

def add_propositions(action: MoveAction, propositions_tuple: Tuple[Set[namedtuple],...]) -> Tuple[Set[namedtuple],...]:
    """Return a set of propositions that are added by the given action. Do not remove any propositions."""
    on_props, on_table_props, clear_props, empty_props, in_stack_props = propositions_tuple
    new_on_props: Set[namedtuple] = on_props.copy()
    new_on_table_props: Set[namedtuple] = on_table_props.copy()
    new_clear_props: Set[namedtuple] = clear_props.copy()
    new_empty_props: Set[namedtuple] = empty_props.copy()
    new_in_stack_props: Set[namedtuple] = in_stack_props.copy()

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


def get_actions(propositions_tuple: Tuple[Set[namedtuple],...]) -> Set[MoveAction]:
    """Return a set of actions that can be applied to the current state."""
    on_props, on_table_props, clear_props, empty_props, in_stack_props = propositions_tuple
    actions = set()

    # Get all blocks and stacks
    blocks = {prop.block for prop in in_stack_props}
    stacks = {prop.stack for prop in in_stack_props}.union({prop.stack for prop in empty_props})

    for block in blocks:
        if ClearProp(block) in clear_props:
            from_stack = next(prop.stack for prop in in_stack_props if prop.block == block)

            # Determine the block below (if any)
            below_block_old = next((prop.below_block for prop in on_props if prop.above_block == block), None)

            for to_stack in stacks:
                if to_stack != from_stack:
                    # Move to an empty stack
                    if EmptyProp(to_stack) in empty_props:
                        action = MoveAction(block, below_block_old, from_stack, to_stack, None)
                        if check_move_action_validity(action, propositions_tuple):
                            actions.add(action)

                    # Move on top of another block
                    for below_block_new in blocks:
                        if ClearProp(below_block_new) in clear_props and below_block_new != block:
                            if InStackProp(below_block_new, to_stack) in in_stack_props:
                                action = MoveAction(block, below_block_old, from_stack, to_stack, below_block_new)
                                if check_move_action_validity(action, propositions_tuple):
                                    actions.add(action)

    return actions



class BlockWorldPlanningProblem:
    def __init__(self, initial_state: BlockWorldState, goal_state: BlockWorldState):
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.num_blocks = initial_state.num_blocks
        self.num_stacks = initial_state.num_stacks

    def get_initial_propositions(self) -> Tuple[Set[namedtuple],...]:
        """Return a set of propositions representing the initial state."""
        return get_propositions(self.initial_state)

    def get_goal_propositions(self) -> Tuple[Set[namedtuple],...]:
        """Return a set of propositions representing the goal state."""
        return get_propositions(self.goal_state)

    def check_goal_propositions(self, propositions: Tuple[Set[namedtuple],...]) -> bool:
        """Return True if the given propositions satisfy the goal state."""
        goal_props = self.get_goal_propositions()
        goal_on_props, goal_on_table_props, goal_clear_props, goal_empty_props, goal_in_stack_props = goal_props
        on_props, on_table_props, clear_props, empty_props, in_stack_props = propositions

        return (goal_on_props.issubset(on_props)
                and goal_on_table_props.issubset(on_table_props)
                and goal_clear_props.issubset(clear_props)
                and goal_empty_props.issubset(empty_props)
                and goal_in_stack_props.issubset(in_stack_props))

def build_relaxed_planning_graph(problem: BlockWorldPlanningProblem) -> Tuple[
    List[Tuple[Set[namedtuple],...]], List[Set[MoveAction]]]:
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

from typing import Dict, Set, Tuple
from collections import defaultdict
import math

def h_max(block_world_state, block_world_goal_state):
    """
    Compute the h_max heuristic for the given state in the Block World problem.
    """
    problem = BlockWorldPlanningProblem(block_world_state, block_world_goal_state)
    goal_props = problem.get_goal_propositions()
    prop_layers, action_layers = build_relaxed_planning_graph(problem)

    prop_costs = [{prop: 0 for prop in set.union(*prop_layers[0])}]
    action_costs = []

    for idx, (action_layer, prop_layer) in enumerate(zip(action_layers, prop_layers[1:])):
        action_costs_dict = dict()
        for action in action_layer:
            preconditions = get_action_preconditions(action, prop_layers[idx])
            action_costs_dict[action] = max(prop_costs[-1][p] for p in preconditions) + 1
        action_costs.append(action_costs_dict)

        prop_costs_dict = dict()
        for prop in set.union(*prop_layer):
            if prop in prop_costs[-1]:
                prop_costs_dict[prop] = prop_costs[-1][prop]
            else:
                action_achievers = get_prop_achievers(prop, action_layer)
                prop_costs_dict[prop] = min(action_costs_dict[action] for action in action_achievers)

        prop_costs.append(prop_costs_dict)

    return max(prop_costs[-1][prop] for prop in set.union(*goal_props))



def get_action_preconditions(action: MoveAction, state: Tuple[Set[namedtuple],...]) -> Set[namedtuple]:
    """Return the set of propositions that are preconditions for the given action."""
    on_props, on_table_props, clear_props, empty_props, in_stack_props = state
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

def get_prop_achievers(prop: namedtuple, actions: Set[MoveAction]) -> Set[MoveAction]:
    """Return the set of actions that achieve the given proposition."""
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

def get_on_prop_achievers(on_prop: OnProp, actions: Set[MoveAction]) -> Set[MoveAction]:
    """Return the set of actions that achieve the given OnProp proposition."""
    return {action for action in actions if action.block == on_prop.above_block and action.below_block_new == on_prop.below_block}

def get_on_table_prop_achievers(on_table_prop: OnTableProp, actions: Set[MoveAction]) -> Set[MoveAction]:
    """Return the set of actions that achieve the given OnTableProp proposition."""
    return {action for action in actions if action.block == on_table_prop.block and action.below_block_new is None}

def get_clear_prop_achievers(clear_prop: ClearProp, actions: Set[MoveAction]) -> Set[MoveAction]:
    """Return the set of actions that achieve the given ClearProp proposition."""
    return {action for action in actions if action.below_block_old == clear_prop.block}

def get_empty_prop_achievers(empty_prop: EmptyProp, actions: Set[MoveAction]) -> Set[MoveAction]:
    """Return the set of actions that achieve the given EmptyProp proposition."""
    return {action for action in actions if action.from_stack == empty_prop.stack and action.below_block_old is None}

def get_in_stack_prop_achievers(in_stack_prop: InStackProp, actions: Set[MoveAction]) -> Set[MoveAction]:
    """Return the set of actions that achieve the given InStackProp proposition."""
    return {action for action in actions if action.block == in_stack_prop.block and action.to_stack == in_stack_prop.stack}