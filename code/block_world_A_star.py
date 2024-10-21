from typing import List, Tuple, Optional, Callable, Dict
import heapq
import os
import pickle
from tqdm import tqdm

from block_world_generator import BlockWorldState, generate_block_world_problem
from block_world_heuristics import misplaced_blocks, height_difference, h_max

class SearchNode:
    def __init__(self, state: BlockWorldState, serial_number: int, g: int, h: int, h_0: int, parent: Optional['SearchNode'] = None,
                 action: Optional[Tuple[int, int]] = None):
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

def a_star(initial_state: BlockWorldState,
           goal_state: BlockWorldState,
           heuristic: Callable[[BlockWorldState, BlockWorldState], int]) -> Tuple[
    Optional[List[Tuple[int, int]]], SearchNode]:

    root_h = heuristic(initial_state, goal_state)
    root = SearchNode(initial_state, 0, 0, root_h, root_h)
    open_set = []
    closed_set = set()
    node_dict: Dict[BlockWorldState, SearchNode] = {initial_state: root}

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

def reconstruct_path(node: SearchNode) -> List[Tuple[int, int]]:
    path = []
    while node.parent:
        path.append(node.action)
        node = node.parent
    return path[::-1]

def calculate_progress(root: SearchNode):
    """Calculate the progress of each node in the search tree."""
    def count_nodes(node: SearchNode) -> int:
        return 1 + sum(count_nodes(child) for child in node.children)

    total_nodes = count_nodes(root)

    def update_progress(node: SearchNode):
        node.progress = node.serial_number / total_nodes
        for child in node.children:
            update_progress(child)

    update_progress(root)

NUM_BLOCKS_LIST = [5, 10, 15, 20]
NUM_STACKS_LIST = [3, 5, 7]
NUM_MOVES_LIST = [5, 10, 15, 20]
SAMPLES = 1000

def main():
    for NUM_BLOCKS in NUM_BLOCKS_LIST:
        for NUM_STACKS in NUM_STACKS_LIST:
            for NUM_MOVES in NUM_MOVES_LIST:
                print(f"Generating samples for {NUM_BLOCKS} blocks, {NUM_STACKS} stacks, {NUM_MOVES} moves")
                for sample_idx in tqdm(range(SAMPLES)):
                    initial_state, goal_state = generate_block_world_problem(NUM_BLOCKS, NUM_STACKS, NUM_MOVES)
                    solution, search_tree_root = a_star(initial_state, goal_state, misplaced_blocks)

                    #prints for debugging
                    # print(f"Initial state:\n{initial_state}")
                    # print(f"Goal state:\n{goal_state}")
                    # print(f"Solution: {solution}")

                    # Calculate progress for each node
                    calculate_progress(search_tree_root)

                    # Create directory if it doesn't exist
                    output_dir = f"dataset/bw_hmax_blocks_{NUM_BLOCKS}_stacks_{NUM_STACKS}_moves_{NUM_MOVES}"
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    # Save the search tree
                    with open(f"{output_dir}/sample_{sample_idx}.pkl", "wb") as f:
                        pickle.dump(search_tree_root, f)

                    # Optional: Print some information about the solution
                    # if solution:
                    #     print(f"Sample {sample_idx}: Solution found with {len(solution)} moves")
                    # else:
                    #     print(f"Sample {sample_idx}: No solution found")

if __name__ == "__main__":
    main()