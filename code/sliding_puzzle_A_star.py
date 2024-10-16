from typing import List, Tuple, Optional, Callable, Dict
import heapq
import pickle
from sliding_puzzle_generator import SlidingPuzzleState, generate_sliding_puzzle_problem
from sliding_puzzle_heuristics import manhattan_distance, misplaced_tiles, h_max, h_ff

class SearchNode:
    def __init__(self, state: SlidingPuzzleState, serial_number: int, g: int, h: int, h_0: int, parent: Optional['SearchNode'] = None,
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

def a_star(initial_state: SlidingPuzzleState,
           goal_state: SlidingPuzzleState,
           heuristic: Callable[[SlidingPuzzleState, SlidingPuzzleState], int]) -> Tuple[
    Optional[List[str]], SearchNode]:
    
    root_h = heuristic(initial_state, goal_state)
    root = SearchNode(initial_state, 0, 0, root_h, root_h)
    open_set = []
    closed_set = set()
    node_dict: Dict[SlidingPuzzleState, SearchNode] = {initial_state: root}

    heapq.heappush(open_set, (root.f, id(root), root))

    serial_number = 0
    global_min_h = root.h
    global_max_f = root.f
    nodes_since_global_min_h = 0
    nodes_since_global_max_f = 0

    while open_set:
        _, _, current_node = heapq.heappop(open_set)

        if current_node.state in closed_set:
            continue  # Skip this node if it's a duplicate

        if current_node.h < global_min_h:
            global_min_h = current_node.h
            nodes_since_global_min_h = 0
        else:
            nodes_since_global_min_h += 1

        if current_node.f > global_max_f:
            global_max_f = current_node.f
            nodes_since_global_max_f = 0
        else:
            nodes_since_global_max_f += 1

        current_node.min_h_seen = global_min_h
        current_node.nodes_since_min_h = nodes_since_global_min_h
        current_node.max_f_seen = global_max_f
        current_node.nodes_since_max_f = nodes_since_global_max_f

        if current_node.state == goal_state:
            return reconstruct_path(current_node), root

        closed_set.add(current_node.state)

        for action in current_node.state.get_possible_actions():
            neighbor_state = current_node.state.apply_action(action)
            if neighbor_state in closed_set:
                continue

            neighbor_g = current_node.g + 1
            neighbor_h = heuristic(neighbor_state, goal_state)

            if neighbor_state not in node_dict or neighbor_g < node_dict[neighbor_state].g:
                serial_number += 1
                neighbor_node = SearchNode(neighbor_state, serial_number, neighbor_g, neighbor_h, root_h, current_node, action)
                node_dict[neighbor_state] = neighbor_node
                current_node.children.append(neighbor_node)
                current_node.child_count += 1

                heapq.heappush(open_set, (neighbor_node.f, id(neighbor_node), neighbor_node))

    return None, root


def reconstruct_path(node: SearchNode) -> List[str]:
    path = []
    while node.parent:
        path.append(node.action)
        node = node.parent
    return path[::-1]


def print_search_tree(node: SearchNode, depth: int = 0):
    indent = "  " * depth
    # print(f"{indent}State:\n{indent}{node.state}")
    print(f"{indent}Serial: {node.serial_number}")
    print(f"{indent}g: {node.g}, h: {node.h}, f: {node.f}")
    print(f"{indent}Child count: {node.child_count}")
    print(f"{indent}Min h seen: {node.min_h_seen}, Nodes since min h: {node.nodes_since_min_h}")
    print(f"{indent}Max f seen: {node.max_f_seen}, Nodes since max f: {node.nodes_since_max_f}")
    print(f"{indent}Progress: {node.progress}")
    for child in node.children:
        print_search_tree(child, depth + 1)


def calculate_progress(root: SearchNode):
    """Calculate the progress of each node in the search tree.

    Args:
        root (SearchNode): The root node of the search tree.
    """
    def count_nodes(node: SearchNode) -> int:
        return 1 + sum(count_nodes(child) for child in node.children)

    total_nodes = count_nodes(root)

    def update_progress(node: SearchNode):
        node.progress = node.serial_number / total_nodes
        for child in node.children:
            update_progress(child)

    update_progress(root)


def main():
    size = 5
    num_moves = 10  # Reduced for a smaller search tree
    initial_state, goal_state = generate_sliding_puzzle_problem(size, num_moves)

    print("Initial State:")
    print(initial_state)
    print("\nGoal State:")
    print(goal_state)

    solution, search_tree_root = a_star(initial_state, goal_state, h_max)

    # Calculate progress for each node
    calculate_progress(search_tree_root)

    if solution:
        print(f"\nSolution found in {len(solution)} moves:")
        print(" -> ".join(solution))
    else:
        print("\nNo solution found.")

    print("\nSearch Tree:")
    print_search_tree(search_tree_root)
    
    with open("search_tree.pkl", "wb") as f:
        pickle.dump(search_tree_root, f)


if __name__ == "__main__":
    main()