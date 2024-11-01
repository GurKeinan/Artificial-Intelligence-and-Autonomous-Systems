""" This module contains the definition of the StateInterface and SearchNode classes. """

from typing import List, Tuple, Optional, Callable, Dict

class StateInterface:
    """
    This class defines the interface for a state in a search problem. It is used by the search algorithms to interact
    """
    def get_possible_actions(self) -> List[str]:
        raise NotImplementedError

    def apply_action(self, action: str) -> 'StateInterface':
        raise NotImplementedError

    def __eq__(self, other: object) -> bool:
        raise NotImplementedError

    def __hash__(self) -> int:
        raise NotImplementedError


class SearchNode:
    """
    This class represents a node in the search tree. It contains the state, the cost of the path to reach this node, the heuristic value of the node, the parent node, the action that led to this node, and the children of this node. It also contains some additional information that is used by the search algorithms to optimize the search process.

    Attributes:
        state (StateInterface): The state of the node.
        g (int): The cost of the path from the root to this node.
        h (int): The heuristic value of the node.
        h_0 (int): The heuristic value of the node with respect to the initial state.
        f (int): The sum of the cost and the heuristic value of the node.
        parent (Optional[SearchNode]): The parent node of this node.
        action (Optional[str]): The action that led to this node.
        children (List[SearchNode]): The children of this node.
        serial_number (int): A unique identifier for this node.
        child_count (int): The number of children of this node.
        min_h_seen (int): The minimum heuristic value seen in the subtree rooted at this node.
        nodes_since_min_h (int): The number of nodes that have been expanded since the minimum heuristic value was last updated.
        max_f_seen (int): The maximum f value seen in the subtree rooted at this node.
        nodes_since_max_f (int): The number of nodes that have been expanded since the maximum f value was last updated.
    """
    def __init__(self, state: StateInterface, serial_number: int, g: int, h: int, h_0: int, parent: Optional['SearchNode'] = None,
                 action: Optional[str] = None):
        self.state = state
        self.g = g
        self.h = h
        self.h_0 = h_0
        self.f = g + h
        self.parent = parent
        self.action = action
        self.children: List['SearchNode'] = []
        self.serial_number: int = serial_number
        self.child_count: int = 0
        self.min_h_seen: int = h
        self.nodes_since_min_h: int = 0
        self.max_f_seen: int = self.f
        self.nodes_since_max_f: int = 0

    def __lt__(self, other: 'SearchNode') -> bool:
        return self.f < other.f

