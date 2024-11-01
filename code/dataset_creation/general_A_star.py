""" This script generates search trees for the sliding puzzle and block world problems using A* search. The search trees are saved as pickle files in the dataset folder. """

from typing import List, Tuple, Optional, Callable, Dict
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


def a_star(initial_state, goal_state, heuristic):
    """
    Perform A* search algorithm to find the shortest path from the initial state to the goal state.

    Args:
        initial_state (StateInterface): The initial state of the problem.
        goal_state (StateInterface): The goal state of the problem.
        heuristic (Callable[[StateInterface, StateInterface], float]): A heuristic function that estimates the cost from the current state to the goal state.

    Returns:
        Tuple[Optional[List[str]], SearchNode]: A tuple containing the list of actions to reach the goal state and the root of the search tree. If no solution is found, the list of actions will be None.
    """

    root_h = heuristic(initial_state, goal_state)
    root = SearchNode(initial_state, 0, 0, root_h, root_h)
    root.min_h_seen = root_h
    root.max_f_seen = root.f
    root.nodes_since_min_h = 0
    root.nodes_since_max_f = 0

    open_set = []
    closed_set = set()
    node_dict: Dict[StateInterface, SearchNode] = {initial_state: root}

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

        if current_node.state == goal_state:
            return reconstruct_path(current_node), root

        closed_set.add(current_node.state)

        for action in current_node.state.get_possible_actions():
            neighbor_state = current_node.state.apply_action(action)
            if neighbor_state in closed_set:
                continue

            neighbor_g = current_node.g + 1
            neighbor_h = heuristic(neighbor_state, goal_state)
            neighbor_f = neighbor_g + neighbor_h

            if neighbor_state not in node_dict or neighbor_g < node_dict[neighbor_state].g:
                # ** Update counters here: **
                serial_number += 1
                nodes_since_global_min_h += 1
                nodes_since_global_max_f += 1

                # ** Create new node: **
                neighbor_node = SearchNode(
                    neighbor_state, serial_number, neighbor_g, neighbor_h, root_h, current_node, action)

                # ** Update global if this new node has a smaller h or larger f: **
                if neighbor_h < global_min_h:
                    global_min_h = neighbor_h
                    nodes_since_global_min_h = 0
                # else:
                #     nodes_since_global_min_h += 1

                if neighbor_f > global_max_f:
                    global_max_f = neighbor_f
                    nodes_since_global_max_f = 0
                # else:
                #     nodes_since_global_max_f += 1

                # ** Set values for the new node to the current global: **
                neighbor_node.min_h_seen = global_min_h
                neighbor_node.max_f_seen = global_max_f
                neighbor_node.nodes_since_min_h = nodes_since_global_min_h
                neighbor_node.nodes_since_max_f = nodes_since_global_max_f

                node_dict[neighbor_state] = neighbor_node
                current_node.children.append(neighbor_node)
                current_node.child_count += 1

                heapq.heappush(open_set, (neighbor_node.f,
                               id(neighbor_node), neighbor_node))

    return None, root


def reconstruct_path(node):
    """
    Reconstruct the path from the initial state to the goal state by following parent pointers.

    Args:
        node (SearchNode): The node representing the goal state.

    Returns:
        List[str]: A list of actions to reach the goal state from the initial state.
    """
    path = []
    while node.parent:
        path.append(node.action)
        node = node.parent
    return path[::-1]


def print_search_tree(node: SearchNode, depth: int = 0):
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


def print_nodes_by_serial_order(node):
    """
    Print all nodes in the search tree in the order of their serial numbers.

    Args:
        node (SearchNode): The root node of the search tree.
    """
    all_nodes = []

    def traverse(node):
        """ Traverse the search tree and add all nodes to the list. """
        all_nodes.append(node)
        for child in node.children:
            traverse(child)

    traverse(node)
    all_nodes.sort(key=lambda n: n.serial_number)

    for node in all_nodes:
        print(f"Serial: {node.serial_number}, Parent Serial: {node.parent.serial_number if node.parent else None}, g: {node.g}, h: {node.h}, f: {node.f}, child_count: {node.child_count}, h_0: {node.h_0}, min_h_seen: {node.min_h_seen}, nodes_since_min_h: {node.nodes_since_min_h}, max_f_seen: {node.max_f_seen}, nodes_since_max_f: {node.nodes_since_max_f}\n")
        print(node.state)
        print("\n")


def calculate_progress(root: SearchNode):
    """
    Calculate the progress of each node in the search tree.

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

def save_sp_search_tree(heuristic_func):
    """
    Save the search tree for the sliding puzzle problem using the specified heuristic function.

    Args:
        heuristic_func (Callable[[StateInterface, StateInterface], float]): The heuristic function to use for the A* search.
    """
    heuristic_name = heuristic_func.__name__
    for SIZE in SIZE_LIST:
        for NUM_MOVES in NUM_MOVES_LIST:
            print(f"Generating search trees for size {SIZE} and {NUM_MOVES} moves...")
            for sample_idx in tqdm(range(SAMPLES)):
                initial_state, goal_state = generate_sliding_puzzle_problem(SIZE, NUM_MOVES)
                solution, search_tree_root = a_star(initial_state, goal_state, sp_h_max)

                # Calculate progress for each node
                calculate_progress(search_tree_root)

                ### Save the search tree: ###
                if not os.path.exists(f"{base_dir}/dataset/{heuristic_name}_size_{SIZE}_moves_{NUM_MOVES}"):
                    os.makedirs(
                        f"{base_dir}/dataset/{heuristic_name}_size_{SIZE}_moves_{NUM_MOVES}")

                with open(f"{base_dir}/dataset/{heuristic_name}_size_{SIZE}_moves_{NUM_MOVES}/sample_{sample_idx}.pkl", "wb") as f:
                    pickle.dump(search_tree_root, f)

def save_bw_search_tree(heuristic_func):
    """
    Save the search tree for the block world problem using the specified heuristic function.

    Args:
        heuristic_func (Callable[[StateInterface, StateInterface], float]): The heuristic function to use for the A* search.
    """
    heuristic_name = heuristic_func.__name__
    for NUM_BLOCKS in NUM_BLOCKS_LIST:
        for NUM_STACKS in NUM_STACKS_LIST:
            for NUM_MOVES in NUM_MOVES_LIST:
                print(f"Generating samples for {NUM_BLOCKS} blocks, {NUM_STACKS} stacks, {NUM_MOVES} moves")
                for sample_idx in tqdm(range(SAMPLES)):
                    initial_state, goal_state = generate_block_world_problem(NUM_BLOCKS, NUM_STACKS, NUM_MOVES)
                    solution, search_tree_root = a_star(initial_state, goal_state, heuristic_func)

                    # Calculate progress for each node
                    calculate_progress(search_tree_root)

                    # Create directory if it doesn't exist
                    output_dir = f"{base_dir}/dataset/{heuristic_name}_blocks_{NUM_BLOCKS}_stacks_{NUM_STACKS}_moves_{NUM_MOVES}"
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

base_dir = Path(__file__).resolve().parent.parent
if base_dir.name != "code":
    base_dir = base_dir / "code"


def main():

    save_sp_search_tree(sp_manhattan_distance)
    save_sp_search_tree(sp_misplaced_tiles)
    save_sp_search_tree(sp_h_max)

    save_bw_search_tree(bw_misplaced_blocks)
    save_bw_search_tree(bw_height_difference)
    save_bw_search_tree(bw_h_max)


if __name__ == "__main__":
    main()
