from typing import List, Tuple, Optional, Callable, Dict

class StateInterface:
    def get_possible_actions(self) -> List[str]:
        raise NotImplementedError

    def apply_action(self, action: str) -> 'StateInterface':
        raise NotImplementedError

    def __eq__(self, other: object) -> bool:
        raise NotImplementedError

    def __hash__(self) -> int:
        raise NotImplementedError
    

class SearchNode:
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

