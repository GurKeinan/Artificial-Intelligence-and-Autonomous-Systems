"""
This module contains the StateInterface and SearchNode classes,
which are used to represent general states and nodes in a search problem.
"""

from typing import List, Optional

class StateInterface:
    """
    Interface for a state in a search problem.
    """
    def get_possible_actions(self) -> List[str]:
        """ Return a list of possible actions that can be applied to the current state. """
        raise NotImplementedError

    def apply_action(self, action: str) -> 'StateInterface':
        """ Apply the given action to the current state and return the new state. """
        raise NotImplementedError

    def __eq__(self, other: object) -> bool:
        """ Check if the current state is equal to another state. """
        return NotImplemented

    def __hash__(self) -> int:
        """ Return a hash value for the current state. """
        raise NotImplementedError


class SearchNode:
    """
    A class used to represent a node in a search tree.
    Attributes
    ----------
    state : StateInterface
        The state associated with this node.
    g : int
        The cost to reach this node from the start node.
    h : int
        The estimated cost to reach the goal from this node.
    h_0 : int
        The initial heuristic value.
    f : int
        The total estimated cost (g + h).
    parent : Optional[SearchNode]
        The parent node of this node.
    action : Optional[str]
        The action taken to reach this node from the parent node.
    children : List[SearchNode]
        The list of child nodes.
    serial_number : int
        A unique identifier for this node.
    child_count : int
        The number of children this node has.
    min_h_seen : int
        The minimum heuristic value seen so far.
    nodes_since_min_h : int
        The number of nodes generated since the minimum heuristic value was seen.
    max_f_seen : int
        The maximum f value seen so far.
    nodes_since_max_f : int
        The number of nodes generated since the maximum f value was seen.
    progress : float
        The progress metric for this node.
    Methods
    -------
    __init__(self, state: StateInterface, serial_number: int, g: int, h: int,
    h_0: int, parent: Optional['SearchNode'] = None, action: Optional[str] = None)
        Initializes a SearchNode with the given parameters.
    __lt__(self, other: 'SearchNode') -> bool
        Compares this node with another node based on their f values.
    """

    def __init__(self, state: StateInterface, serial_number: int, g: int, h: int,
                 h_0: int, parent: Optional['SearchNode'] = None, action: Optional[str] = None):
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
        self.progress: float = 0.0

    def __lt__(self, other: 'SearchNode') -> bool:
        return self.f < other.f
