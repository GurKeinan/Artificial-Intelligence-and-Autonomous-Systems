""" This script generates the dataset for the sliding puzzle and block world problems using parallel processing."""

from pathlib import Path
import pickle
from tqdm import tqdm
import multiprocessing as mp

from sliding_puzzle_generator import generate_sliding_puzzle_problem
from block_world_generator import generate_block_world_problem
from sliding_puzzle_heuristics import sp_manhattan_distance, sp_misplaced_tiles, sp_h_max
from block_world_heuristics import bw_misplaced_blocks, bw_height_difference, bw_h_max

# Import functions from general_A_star.py
from general_a_star import a_star, calculate_progress

def process_single_sp_problem(args):
    """Worker function to process a single sliding puzzle problem instance"""
    size, num_moves, sample_idx, worker_id, heuristic_func = args

    try:
        initial_state, goal_state = generate_sliding_puzzle_problem(size, num_moves)
        solution, search_tree_root = a_star(initial_state, goal_state, heuristic_func)

        # Calculate progress for each node
        calculate_progress(search_tree_root)

        return {
            'tree': search_tree_root,
            'size': size,
            'num_moves': num_moves,
            'sample_idx': sample_idx,
            'worker_id': worker_id,
            'heuristic_name': heuristic_func.__name__
        }
    except Exception as e:
        print(f"Error processing sliding puzzle problem: {e}")
        return None

def process_single_bw_problem(args):
    """Worker function to process a single block world problem instance"""
    num_blocks, num_stacks, num_moves, sample_idx, worker_id, heuristic_func = args

    try:
        initial_state, goal_state = generate_block_world_problem(num_blocks, num_stacks, num_moves)
        solution, search_tree_root = a_star(initial_state, goal_state, heuristic_func)

        # Calculate progress for each node
        calculate_progress(search_tree_root)

        return {
            'tree': search_tree_root,
            'num_blocks': num_blocks,
            'num_stacks': num_stacks,
            'num_moves': num_moves,
            'sample_idx': sample_idx,
            'worker_id': worker_id,
            'heuristic_name': heuristic_func.__name__
        }
    except Exception as e:
        print(f"Error processing block world problem: {e}")
        return None

def save_sp_results(result, base_dir):
    """Save the sliding puzzle search tree with proper naming"""
    if result is None:
        return

    heuristic_name = result['heuristic_name']
    size = result['size']
    num_moves = result['num_moves']
    sample_idx = result['sample_idx']
    worker_id = result['worker_id']

    output_dir = Path(base_dir) / "dataset" / f"sp_{heuristic_name}_size_{size}_moves_{num_moves}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use both sample_idx and worker_id to ensure unique filenames
    output_file = output_dir / f"sample_{sample_idx}_worker_{worker_id}.pkl"

    with open(output_file, 'wb') as f:
        pickle.dump(result['tree'], f)

def save_bw_results(result, base_dir):
    """Save the block world search tree with proper naming"""
    if result is None:
        return

    heuristic_name = result['heuristic_name']
    num_blocks = result['num_blocks']
    num_stacks = result['num_stacks']
    num_moves = result['num_moves']
    sample_idx = result['sample_idx']
    worker_id = result['worker_id']

    output_dir = Path(base_dir) / "dataset" / f"bw_{heuristic_name}_blocks_{num_blocks}_stacks_{num_stacks}_moves_{num_moves}"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"sample_{sample_idx}_worker_{worker_id}.pkl"

    with open(output_file, 'wb') as f:
        pickle.dump(result['tree'], f)

def save_sp_search_tree_parallel(heuristic_func):
    """Parallel version of save_sp_search_tree"""
    print(f"Starting parallel processing for {heuristic_func.__name__}")

    # Create all parameter combinations
    all_params = []
    worker_id = 0
    for size in SIZE_LIST:
        for num_moves in NUM_MOVES_LIST:
            for sample_idx in range(SAMPLES):
                all_params.append((size, num_moves, sample_idx, worker_id, heuristic_func))
                worker_id += 1

    # Determine number of processes - use 80% of available CPUs
    num_processes = max(1, int(mp.cpu_count() * 0.8))

    # Create the process pool and run the jobs
    with mp.Pool(processes=num_processes) as pool:
        results = []
        for result in tqdm(pool.imap_unordered(process_single_sp_problem, all_params),
                         total=len(all_params),
                         desc=f"Processing Sliding Puzzle - {heuristic_func.__name__}"):
            if result:
                save_sp_results(result, base_dir)
                results.append(result)

    return results

def save_bw_search_tree_parallel(heuristic_func):
    """Parallel version of save_bw_search_tree"""
    print(f"Starting parallel processing for {heuristic_func.__name__}")

    # Create all parameter combinations
    all_params = []
    worker_id = 0
    for num_blocks in NUM_BLOCKS_LIST:
        for num_stacks in NUM_STACKS_LIST:
            for num_moves in NUM_MOVES_LIST:
                for sample_idx in range(SAMPLES):
                    all_params.append((num_blocks, num_stacks, num_moves, sample_idx, worker_id, heuristic_func))
                    worker_id += 1

    # Determine number of processes - use 80% of available CPUs
    num_processes = max(1, int(mp.cpu_count() * 0.8))

    # Create the process pool and run the jobs
    with mp.Pool(processes=num_processes) as pool:
        results = []
        for result in tqdm(pool.imap_unordered(process_single_bw_problem, all_params),
                         total=len(all_params),
                         desc=f"Processing Block World - {heuristic_func.__name__}"):
            if result:
                save_bw_results(result, base_dir)
                results.append(result)

    return results

# Constants
# Sliding Puzzle:
SIZE_LIST = [5, 7, 9]

# Block World:
NUM_BLOCKS_LIST = [5, 10]
NUM_STACKS_LIST = [3, 5]

# Problem Settings:
NUM_MOVES_LIST = [7, 10, 15]
SAMPLES = 200

# Set up base directory
base_dir = Path(__file__).resolve().parent
if base_dir.name != "code":
    base_dir = base_dir / "code"

def main():
    # Create base directory if it doesn't exist
    Path(base_dir).mkdir(parents=True, exist_ok=True)

    print(f"Using {max(1, int(mp.cpu_count() * 0.8))} processes")

    # Run the sliding puzzle generators in parallel
    sp_heuristic_functions = [sp_manhattan_distance, sp_misplaced_tiles, sp_h_max]
    for heuristic_func in sp_heuristic_functions:
        save_sp_search_tree_parallel(heuristic_func)

    # Run the block world generators in parallel
    bw_heuristic_functions = [bw_misplaced_blocks, bw_height_difference, bw_h_max]
    for heuristic_func in bw_heuristic_functions:
        save_bw_search_tree_parallel(heuristic_func)

    print("All processing complete!")

if __name__ == "__main__":
    # Ensure proper multiprocessing behavior on Windows
    mp.freeze_support()
    main()