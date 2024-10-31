"""
    This module implements the A* search algorithm and related utilities
    for solving sliding puzzle and block world problems.

Functions:
    a_star(initial_state: StateInterface, goal_state: StateInterface,
    heuristic: Callable[[StateInterface, StateInterface], int])
    -> Tuple[Optional[List[str]], SearchNode]:
        Perform A* search to find the optimal path from the initial state to the goal state
        using the provided heuristic function.

    reconstruct_path(node: SearchNode) -> List[str]:
        Reconstruct the path from the initial state to the goal state
        by tracing back from the given search node.

    print_search_tree(node: SearchNode, depth: int = 0):
        Print the search tree starting from the given node,
        displaying state information and search metrics.

    print_nodes_by_serial_order(node: SearchNode):
        Print all nodes in the search tree in the order of their serial numbers.

    calculate_progress(root: SearchNode):
        Calculate the progress of each node in the search tree based on its serial number.

    debug_print_search_tree(initial_state, goal_state, solution, search_tree_root):
        Print information including the initial state, goal state, solution path, and search tree.

    save_sp_search_tree(heuristic_func):
        Generate and save search trees for sliding puzzle problems using heuristic function.

    save_bw_search_tree(heuristic_func):
        Generate and save search trees for block world problems using heuristic function.

Constants:
    SIZE_LIST: List[int]
        List of sizes for sliding puzzle problems.

    NUM_BLOCKS_LIST: List[int]
        List of block counts for block world problems.

    NUM_STACKS_LIST: List[int]
        List of stack counts for block world problems.

    NUM_MOVES_LIST: List[int]
        List of move counts for generating problems.

    SAMPLES: int
        Number of samples to generate for each problem configuration.

    base_dir: Path
        Base directory for saving generated datasets.

main():
    Main function to generate and save search trees for sliding puzzle and block world problems
    using various heuristics.
"""

from typing import List, Tuple, Optional, Callable
import heapq
import os
from pathlib import Path
import pickle
from tqdm import tqdm
from general_state import StateInterface, SearchNode
from sliding_puzzle_generator import SlidingPuzzleState, generate_sliding_puzzle_problem
from block_world_generator import BlockWorldState, generate_block_world_problem
from sliding_puzzle_heuristics import sp_manhattan_distance, sp_misplaced_tiles, sp_h_max
from block_world_heuristics import bw_misplaced_blocks, bw_height_difference, bw_h_max


def a_star(initial_state: StateInterface,
           goal_state: StateInterface,
           heuristic: Callable[[StateInterface, StateInterface], int]) -> Tuple[
        Optional[List[str]], SearchNode]:
    """
    Perform the A* algorithm to find the shortest path from the initial state to the goal state.

    :param initial_state: The initial state of the search.
    :param goal_state: The goal state of the search.
    :param heuristic: A heuristic function that estimates the cost
    from the current state to the goal state.
    :return: A tuple containing the list of actions to reach the goal and the root search node.
    """
    root = initialize_root_node(initial_state, goal_state, heuristic)
    open_set, closed_set, node_dict = initialize_search_structures(root)

    global_min_h, global_max_f = root.h, root.f
    nodes_since_global_min_h, nodes_since_global_max_f = 0, 0
    serial_number = 0

    while open_set:
        current_node = get_next_node(open_set, closed_set)
        if current_node is None:
            continue

        if current_node.state == goal_state:
            return reconstruct_path(current_node), root

        closed_set.add(current_node.state)

        for neighbor_node in generate_neighbors(current_node, goal_state, heuristic, node_dict):
            if neighbor_node.state in closed_set:
                continue

            serial_number, nodes_since_global_min_h, nodes_since_global_max_f = update_counters(
                serial_number, nodes_since_global_min_h, nodes_since_global_max_f)

            global_min_h, global_max_f, nodes_since_global_min_h, nodes_since_global_max_f = update_globals(neighbor_node, global_min_h, global_max_f,
                        nodes_since_global_min_h, nodes_since_global_max_f)

            set_node_globals(neighbor_node, global_min_h, global_max_f,
                             nodes_since_global_min_h, nodes_since_global_max_f)

            node_dict[neighbor_node.state] = neighbor_node
            current_node.children.append(neighbor_node)
            current_node.child_count += 1

            heapq.heappush(open_set, (neighbor_node.f, id(neighbor_node), neighbor_node))

    return None, root

def initialize_root_node(initial_state, goal_state, heuristic):
    """
    Initialize the root node of the search tree.
    """
    root_h = heuristic(initial_state, goal_state)
    root = SearchNode(initial_state, 0, 0, root_h, root_h)
    root.min_h_seen = root_h
    root.max_f_seen = root.f
    root.nodes_since_min_h = 0
    root.nodes_since_max_f = 0
    return root

def initialize_search_structures(root):
    """ Initialize the search structures for A* search. """
    open_set = []
    closed_set = set()
    node_dict = {root.state: root}
    heapq.heappush(open_set, (root.f, id(root), root))
    return open_set, closed_set, node_dict

def get_next_node(open_set, closed_set):
    """ Get the next node from the open set. """
    _, _, current_node = heapq.heappop(open_set)
    if current_node.state in closed_set:
        return None
    return current_node

def generate_neighbors(current_node, goal_state, heuristic, node_dict):
    """ Generate neighbor nodes for the current node. """
    neighbors = []
    for action in current_node.state.get_possible_actions():
        neighbor_state = current_node.state.apply_action(action)
        neighbor_g = current_node.g + 1
        neighbor_h = heuristic(neighbor_state, goal_state)
        neighbor_f = neighbor_g + neighbor_h

        if neighbor_state not in node_dict or neighbor_g < node_dict[neighbor_state].g:
            neighbor_node = SearchNode(neighbor_state, current_node.serial_number + 1,
                                       neighbor_g, neighbor_h, neighbor_f, current_node, action)
            neighbors.append(neighbor_node)
    return neighbors

def update_counters(serial_number, nodes_since_global_min_h, nodes_since_global_max_f):
    """ Update the counters for serial number and nodes since global minimum h and maximum f. """
    serial_number += 1
    nodes_since_global_min_h += 1
    nodes_since_global_max_f += 1
    return serial_number, nodes_since_global_min_h, nodes_since_global_max_f

def update_globals(neighbor_node, global_min_h, global_max_f, nodes_since_global_min_h,
                   nodes_since_global_max_f):
    """ Update the global minimum h and maximum f values. """
    if neighbor_node.h < global_min_h:
        global_min_h = neighbor_node.h
        nodes_since_global_min_h = 0

    if neighbor_node.f > global_max_f:
        global_max_f = neighbor_node.f
        nodes_since_global_max_f = 0

    return global_min_h, global_max_f, nodes_since_global_min_h, nodes_since_global_max_f

def set_node_globals(neighbor_node, global_min_h, global_max_f, nodes_since_global_min_h,
                     nodes_since_global_max_f):
    """ Set the global minimum h and maximum f values for the neighbor node. """
    neighbor_node.min_h_seen = global_min_h
    neighbor_node.max_f_seen = global_max_f
    neighbor_node.nodes_since_min_h = nodes_since_global_min_h
    neighbor_node.nodes_since_max_f = nodes_since_global_max_f


def reconstruct_path(node: SearchNode) -> List[str]:
    """
    Reconstructs the path from the given node to the start node.
    This function traces back from the given node to the start node by following the parent
    references and collects the actions taken to reach each node. The path is returned in the
    correct order from the start node to the given node.
    Args:
        node (SearchNode): The node from which to start reconstructing the path.
    Returns:
        List[str]: A list of actions representing the path from the start node to the given node.
    """

    path = []
    while node.parent:
        path.append(node.action)
        node = node.parent
    return path[::-1]


def print_search_tree(node: SearchNode, depth: int = 0):
    """ Print the search tree starting from the given node."""
    indent = "  " * depth
    print(f"{indent}State:\n{indent}{node.state}")
    print(f"{indent}Serial: {node.serial_number}")
    print(f"{indent}g: {node.g}, h: {node.h}, f: {node.f}")
    print(f"{indent}Child count: {node.child_count}")
    print(f"{indent}Min h seen: {node.min_h_seen}, Nodes since min h: {node.nodes_since_min_h}")
    print(f"{indent}Max f seen: {node.max_f_seen}, Nodes since max f: {node.nodes_since_max_f}")
    print(f"{indent}Progress: {node.progress}")
    for child in node.children:
        print_search_tree(child, depth + 1)


def print_nodes_by_serial_order(node: SearchNode):
    """ Print all nodes in the search tree in the order of their serial numbers."""
    all_nodes = []

    def traverse(node):
        all_nodes.append(node)
        for child in node.children:
            traverse(child)

    traverse(node)
    all_nodes.sort(key=lambda n: n.serial_number)

    for node in all_nodes:
        print(f"Serial: {node.serial_number},\
              Parent Serial: {node.parent.serial_number if node.parent else None},\
              g: {node.g},\
              h: {node.h},\
              f: {node.f},\
              child_count: {node.child_count},\
              h_0: {node.h_0},\
              min_h_seen: {node.min_h_seen},\
              nodes_since_min_h: {node.nodes_since_min_h},\
              max_f_seen: {node.max_f_seen},\
              nodes_since_max_f: {node.nodes_since_max_f}\n")
        print(node.state)
        print("\n")


def calculate_progress(root: SearchNode):
    """ Calculate the progress of each node in the search tree based on its serial number."""
    def count_nodes(node: SearchNode) -> int:
        return 1 + sum(count_nodes(child) for child in node.children)

    total_nodes = count_nodes(root)

    def update_progress(node: SearchNode):
        node.progress = node.serial_number / total_nodes
        for child in node.children:
            update_progress(child)

    update_progress(root)


def debug_print_search_tree(initial_state, goal_state, solution, search_tree_root):
    """ Print the initial state, goal state, solution path, and search tree."""
    print("\nInitial State:")
    print(initial_state)
    print("\nGoal State:")
    print(goal_state)

    if solution:
        print(f"\nSolution found in {len(solution)} moves:")
        print(" -> ".join(solution))
    else:
        print("\nNo solution found.")

    print("\nSearch Tree:\n")
    print_search_tree(search_tree_root)

    print("\nNodes by serial order:\n")
    print_nodes_by_serial_order(search_tree_root)


def save_sp_search_tree(heuristic_func):
    """
    Generate and save search trees for sliding puzzle problems using heuristic function.

    Args:
        heuristic_func (Callable[[StateInterface, StateInterface], int]): The heuristic function to
        use for A* search.
    """
    heuristic_name = heuristic_func.__name__
    for size in SIZE_LIST:
        for num_moves in NUM_MOVES_LIST:
            print(f"Generating search trees for size {size} and {num_moves} moves...")
            for sample_idx in tqdm(range(SAMPLES)):
                initial_state, goal_state = generate_sliding_puzzle_problem(size, num_moves)
                _, search_tree_root = a_star(initial_state, goal_state, heuristic_func)

                # Calculate progress for each node
                calculate_progress(search_tree_root)

                # Debug print the search tree: #
                # debug_print_search_tree(initial_state, goal_state, solution, search_tree_root)

                ### Save the search tree: ###
                if not os.path.exists(
                    f"{base_dir}/dataset/{heuristic_name}_size_{size}_moves_{num_moves}"):
                    os.makedirs(
                        f"{base_dir}/dataset/{heuristic_name}_size_{size}_moves_{num_moves}")

                with open(f"{base_dir}/dataset/{heuristic_name}_size_{size}_moves_{num_moves}/sample_{sample_idx}.pkl", "wb") as f:
                    pickle.dump(search_tree_root, f)

def save_bw_search_tree(heuristic_func):
    """
    Generate and save search trees for block world problems using the provided heuristic function.

    Args:
        heuristic_func (Callable[[StateInterface, StateInterface], int]): The heuristic function to
        use for A* search.
    """
    heuristic_name = heuristic_func.__name__
    for num_blocks in NUM_BLOCKS_LIST:
        for num_stacks in NUM_STACKS_LIST:
            for num_moves in NUM_MOVES_LIST:
                print(f"Generating samples for {num_blocks} blocks, {num_stacks} stacks, {num_moves} moves")
                for sample_idx in tqdm(range(SAMPLES)):
                    initial_state, goal_state = generate_block_world_problem(num_blocks, num_stacks, num_moves)
                    _, search_tree_root = a_star(initial_state, goal_state, heuristic_func)

                    # Calculate progress for each node
                    calculate_progress(search_tree_root)

                    # Create directory if it doesn't exist
                    output_dir = f"{base_dir}/dataset/{heuristic_name}_blocks_{num_blocks}_stacks_{num_stacks}_moves_{num_moves}"
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Save the search tree
                    with open(f"{output_dir}/sample_{sample_idx}.pkl", "wb") as f:
                        pickle.dump(search_tree_root, f)

# Sliding Puzzle:
SIZE_LIST = [5, 7]

# Block World:
NUM_BLOCKS_LIST = [5, 10]
NUM_STACKS_LIST = [3, 5]

# Problem Settings:
NUM_MOVES_LIST = [7, 12]
SAMPLES = 50

base_dir = Path(__file__).resolve().parent
if base_dir.name != "code":
    base_dir = base_dir / "code"


def main():
    """
    Generate and save search trees for sliding puzzle and block world problems
    using various heuristics.
    """

    save_sp_search_tree(sp_manhattan_distance)
    save_sp_search_tree(sp_misplaced_tiles)
    save_sp_search_tree(sp_h_max)

    save_bw_search_tree(bw_misplaced_blocks)
    save_bw_search_tree(bw_height_difference)
    save_bw_search_tree(bw_h_max)


if __name__ == "__main__":
    main()
