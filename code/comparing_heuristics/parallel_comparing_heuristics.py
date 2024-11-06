"""
This module provides functionality to compare different heuristics for solving
sliding puzzle and blocks world problems using the A* search algorithm. The
comparison is done in parallel to utilize multiple CPU cores efficiently.
"""

import sys
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple
from pathlib import Path
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
dataset_creation_path = repo_root / "dataset_creation"
sys.path.append(str(dataset_creation_path))

from sliding_puzzle_generator import generate_sliding_puzzle_problem
from block_world_generator import generate_block_world_problem
from general_A_star import a_star
from sliding_puzzle_heuristics import sp_manhattan_distance, sp_misplaced_tiles, sp_h_max
from block_world_heuristics import bw_misplaced_blocks, bw_height_difference, bw_h_max

# create models directory if it doesn't exist
models_dir = repo_root / "models"
models_dir.mkdir(exist_ok=True)

# Create logs directory if it doesn't exist
log_dir = repo_root / "logs"
log_dir.mkdir(exist_ok=True)

# Remove existing log file
log_filename = log_dir / "heuristic_comparison.log"
if log_filename.exists():
    log_filename.unlink()

file_handler = logging.FileHandler(log_filename)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

def solve_single_problem(problem_tuple, heuristics):
    """Worker function to solve a single problem with all heuristics"""
    initial_state, goal_state = problem_tuple
    problem_results = {}
    success = True

    for heur_name, heur_func in heuristics.items():
        solution, search_tree = a_star(initial_state, goal_state, heur_func)

        if solution is None:
            success = False
            break

        problem_results[heur_name] = {
            'search_tree': search_tree,
            'solution_length': len(solution)
        }

    return success, problem_results if success else None

class HeuristicComparison:
    """
    A class to compare different heuristics for problem-solving in parallel.

    Attributes:
        domain (str): The problem domain, either 'sliding_puzzle' or 'blocks_world'.
        max_workers (int): The maximum number of workers for parallel processing.
        problem_generators (dict): A dictionary mapping domains
        to their respective problem generation functions.
        heuristics (dict): A dictionary mapping domains to their respective heuristics.

    Methods:
        generate_problems(num_problems: int, **kwargs) -> List[Tuple]:
            Generate multiple problem instances in parallel.

        solve_with_heuristics(problems: List[Tuple]) -> List[Dict]:
            Solve problems in parallel using all available heuristics.

        _analyze_search_trees(problem_results: Dict) -> Dict:
            Analyze search trees and compute metrics for heuristic comparison.

        _count_nodes(root) -> int:
            Count total nodes in a search tree.

        _get_state_serial_map(root) -> Dict:
            Create mapping of states to their serial numbers.

        run_comparison(num_problems: int, **kwargs):
            Run full comparison and save results.

        run_parameter_study():
            Run parameter study in parallel.

        plot_parameter_study(results):
            Create and save visualizations for parameter study results.
    """

    def __init__(self, domain: str, max_workers: int):
        self.domain = domain
        self.max_workers = max_workers
        self.problem_generators = {
            'sliding_puzzle': generate_sliding_puzzle_problem,
            'blocks_world': generate_block_world_problem
        }
        self.heuristics = {
            'sliding_puzzle': {
                'manhattan': sp_manhattan_distance,
                'misplaced': sp_misplaced_tiles,
                'h_max': sp_h_max
            },
            'blocks_world': {
                'misplaced': bw_misplaced_blocks,
                'height_diff': bw_height_difference,
                'h_max': bw_h_max
            }
        }

    def generate_problems(self, num_problems: int, **kwargs) -> List[Tuple]:
        """Generate multiple problem instances in parallel"""
        if self.domain == 'sliding_puzzle':
            desc = f"Generating sliding puzzle problems - size={kwargs['size']}, num_moves={kwargs['num_moves']}"
        else:
            desc = f"Generating blocks world problems - num_blocks={kwargs['num_blocks']},num_stacks={kwargs['num_stacks']}, num_moves={kwargs['num_moves']}"

        logger.info(desc)

        problems = []
        # Use process pool for problem generation
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for _ in range(num_problems):
                futures.append(executor.submit(
                    self.problem_generators[self.domain], **kwargs))
            for future in tqdm(as_completed(futures), total=num_problems,desc=desc):
                problems.append(future.result())

        logger.info("Generated %d problems successfully", num_problems)
        return problems

    def solve_with_heuristics(self, problems: List[Tuple]) -> List[Dict]:
        """Solve problems in parallel using all available heuristics"""
        logger.info("Solving problems with all available heuristics")
        all_results = []

        # Create partial function with fixed heuristics
        solve_func = partial(solve_single_problem, heuristics=self.heuristics[self.domain])

        # Use process pool for solving problems
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for problem in problems:
                futures.append(executor.submit(solve_func, problem))

            for future in tqdm(as_completed(futures), total=len(problems), desc="Solving problems"):
                success, results = future.result()
                if success:
                    metrics = self._analyze_search_trees(results)
                    all_results.append(metrics)
        logger.info("Solved %d problems successfully", len(all_results))
        return all_results

    def _analyze_search_trees(self, problem_results: Dict) -> Dict:
        metrics = {}
        heuristics = list(problem_results.keys())

        # Compute basic metrics for each heuristic
        for heur in heuristics:
            metrics.update(self._compute_basic_metrics(heur, problem_results[heur]))

        # Check if solution lengths are identical
        metrics['solution_lengths_identical'] = self._check_solution_lengths_identical(
            problem_results)

        # Compare heuristics pairwise
        for i, heur1 in enumerate(heuristics):
            for heur2 in heuristics[i+1:]:
                metrics.update(self._compare_heuristics(heur1, heur2, problem_results))

        return metrics

    def _compute_basic_metrics(self, heur: str, heur_result: Dict) -> Dict:
        metrics = {}
        tree = heur_result['search_tree']
        metrics[f'total_nodes_{heur}'] = self._count_nodes(tree)
        metrics[f'solution_length_{heur}'] = heur_result['solution_length']
        return metrics

    def _check_solution_lengths_identical(self, problem_results: Dict) -> bool:
        lengths = [result['solution_length'] for result in problem_results.values()]
        return len(set(lengths)) == 1

    def _compare_heuristics(self, heur1: str, heur2: str, problem_results: Dict) -> Dict:
        metrics = {}
        tree1 = problem_results[heur1]['search_tree']
        tree2 = problem_results[heur2]['search_tree']

        map1 = self._get_state_serial_map(tree1)
        map2 = self._get_state_serial_map(tree2)

        states1 = set(map1.keys())
        states2 = set(map2.keys())
        shared_states = states1.intersection(states2)
        only_in_1 = states1 - states2
        only_in_2 = states2 - states1

        metrics[f'shared_states_{heur1}_{heur2}'] = len(shared_states)
        metrics[f'unique_to_{heur1}'] = len(only_in_1)
        metrics[f'unique_to_{heur2}'] = len(only_in_2)

        if shared_states:
            max_serial1 = max(map1.values())
            max_serial2 = max(map2.values())

            if max_serial1 > 0 and max_serial2 > 0:
                order_diffs = [
                    abs((map1[state] / max_serial1) - (map2[state] / max_serial2))
                    for state in shared_states
                ]
                metrics[f'avg_order_diff_{heur1}_{heur2}'] = sum(order_diffs) / len(order_diffs)
                metrics[f'max_order_diff_{heur1}_{heur2}'] = max(order_diffs)
            else:
                metrics[f'avg_order_diff_{heur1}_{heur2}'] = -1
                metrics[f'max_order_diff_{heur1}_{heur2}'] = -1
        else:
            metrics[f'avg_order_diff_{heur1}_{heur2}'] = -1
            metrics[f'max_order_diff_{heur1}_{heur2}'] = -1

        return metrics

    def _count_nodes(self, root) -> int:
        """Count total nodes in a search tree"""
        count = 1
        for child in root.children:
            count += self._count_nodes(child)
        return count

    def _get_state_serial_map(self, root) -> Dict:
        """Create mapping of states to their serial numbers"""
        state_map = {}

        def traverse(node):
            state_map[str(node.state)] = node.serial_number
            for child in node.children:
                traverse(child)

        traverse(root)
        return state_map

    def run_comparison(self, num_problems: int, **kwargs):
        """Run full comparison and save results"""
        problems = self.generate_problems(num_problems, **kwargs)
        results = self.solve_with_heuristics(problems)

        if not results:
            print("No problems were solved successfully by all heuristics")
            return None

        df = pd.DataFrame(results)
        return df

    def run_parameter_study(self):
        """Run parameter study in parallel"""
        if self.domain == 'sliding_puzzle':
            params = [
                {'size': 4, 'num_moves': 5},
                {'size': 4, 'num_moves': 8},
                {'size': 4, 'num_moves': 11},
                {'size': 6, 'num_moves': 5},
                {'size': 6, 'num_moves': 8},
                {'size': 6, 'num_moves': 11}
            ]
        else:  # blocks_world
            params = [
                {'num_blocks': 5, 'num_stacks': 3, 'num_moves': 5},
                {'num_blocks': 5, 'num_stacks': 3, 'num_moves': 8},
                {'num_blocks': 5, 'num_stacks': 3, 'num_moves': 11},
                {'num_blocks': 7, 'num_stacks': 4, 'num_moves': 5},
                {'num_blocks': 7, 'num_stacks': 4, 'num_moves': 8},
                {'num_blocks': 7, 'num_stacks': 4, 'num_moves': 11},
                {'num_blocks': 9, 'num_stacks': 5, 'num_moves': 5},
                {'num_blocks': 9, 'num_stacks': 5, 'num_moves': 8},
                {'num_blocks': 9, 'num_stacks': 5, 'num_moves': 11}
            ]

        results = []
        for param_set in tqdm(params, desc="Parameter combinations"):
            result = self.run_comparison(num_problems=50, **param_set)
            if result is not None and not result.empty:
                results.append((param_set, result))

        if not results:
            logger.error("No successful comparisons found")
            return

        self.plot_parameter_study(results)
        logger.info("Parameter study completed successfully")

    def plot_parameter_study(self, results):
        """Create and save visualizations for parameter study results"""
        n_comparisons = len(results)
        if n_comparisons == 0:
            return

        # Create directory if it doesn't exist
        file_dir = Path(__file__).resolve().parent
        plot_dir = file_dir / 'plots'
        plot_dir.mkdir(exist_ok=True)

        # Determine subplot grid dimensions
        n_cols = min(3, n_comparisons)
        n_rows = (n_comparisons + n_cols - 1) // n_cols

        # Plot node counts
        self._plot_node_counts(results, n_rows, n_cols, plot_dir)

        # Plot state space overlap
        self._plot_state_space_overlap(results, n_rows, n_cols, plot_dir)

        # Plot solution lengths if differences exist
        if any(not df['solution_lengths_identical'].all() for _, df in results):
            self._plot_solution_lengths(results, n_rows, n_cols, plot_dir)

        # Plot order differences
        self._plot_order_differences(results, n_rows, n_cols, plot_dir)

        # Save the raw data
        self._save_raw_data(results, file_dir)

    def _plot_node_counts(self, results, n_rows, n_cols, plot_dir):
        plt.figure(figsize=(6*n_cols, 5*n_rows))
        plt.suptitle(f'{self.domain.title()} - Nodes Expanded by Different Heuristics', y=1.02)

        for idx, (params, df) in enumerate(results):
            plt.subplot(n_rows, n_cols, idx+1)
            node_cols = [col for col in df.columns if col.startswith('total_nodes_')]
            df[node_cols].boxplot()

            param_str = '\n'.join(f'{k}={v}' for k, v in params.items())
            plt.title(f'Parameters:\n{param_str}')
            plt.xticks(rotation=45)
            plt.ylabel('Number of Nodes')

        plt.tight_layout()
        plt.savefig(plot_dir / f'{self.domain}_nodes_expanded.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_state_space_overlap(self, results, n_rows, n_cols, plot_dir):
        plt.figure(figsize=(6*n_cols, 5*n_rows))
        plt.suptitle(f'{self.domain.title()} - State Space Overlap Between Heuristics', y=1.02)

        for idx, (params, df) in enumerate(results):
            plt.subplot(n_rows, n_cols, idx+1)

            heuristics = list(self.heuristics[self.domain].keys())
            x_positions = []
            x_labels = []

            for i, h1 in enumerate(heuristics):
                for h2 in heuristics[i+1:]:
                    if f'shared_states_{h1}_{h2}' in df.columns:
                        shared = df[f'shared_states_{h1}_{h2}'].mean()
                        only_h1 = df[f'unique_to_{h1}'].mean()
                        only_h2 = df[f'unique_to_{h2}'].mean()
                        unique_total = only_h1 + only_h2

                        x_pos = len(x_positions)
                        x_positions.append(x_pos)
                        x_labels.append(f'{h1}\nvs\n{h2}')

                        plt.bar([x_pos], [shared], color='royalblue',
                                label='Shared States' if idx == 0 and x_pos == 0 else "")
                        plt.bar([x_pos], [unique_total], bottom=[shared], color='coral',
                                label='Unique States' if idx == 0 and x_pos == 0 else "")

            param_str = '\n'.join(f'{k}={v}' for k, v in params.items())
            plt.title(f'Parameters:\n{param_str}')
            plt.xticks(x_positions, x_labels, rotation=45)
            plt.ylabel('Number of States')

            if idx == 0:
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(plot_dir / f'{self.domain}_state_overlap.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_solution_lengths(self, results, n_rows, n_cols, plot_dir):
        plt.figure(figsize=(6*n_cols, 5*n_rows))
        plt.suptitle(f'{self.domain.title()} - Solution Lengths Across Heuristics', y=1.02)

        for idx, (params, df) in enumerate(results):
            plt.subplot(n_rows, n_cols, idx+1)
            length_cols = [col for col in df.columns if col.startswith('solution_length_')]
            df[length_cols].boxplot()

            non_identical = (~df['solution_lengths_identical']).sum()
            param_str = '\n'.join(f'{k}={v}' for k, v in params.items())

            if non_identical > 0:
                plt.title(f'Parameters:\n{param_str}\n⚠️ {non_identical} cases \
                    with different lengths')
            else:
                plt.title(f'Parameters:\n{param_str}\n(All solutions optimal)')

            plt.xticks(rotation=45)
            plt.ylabel('Solution Length')

        plt.tight_layout()
        plt.savefig(plot_dir / f'{self.domain}_solution_lengths.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_order_differences(self, results, n_rows, n_cols, plot_dir):
        plt.figure(figsize=(6*n_cols, 5*n_rows))
        plt.suptitle(f'{self.domain.title()} - Node Expansion Order Differences', y=1.02)

        for idx, (params, df) in enumerate(results):
            plt.subplot(n_rows, n_cols, idx+1)
            diff_cols = [col for col in df.columns if col.startswith('avg_order_diff_')]
            if diff_cols:
                df[diff_cols].boxplot()

                param_str = '\n'.join(f'{k}={v}' for k, v in params.items())
                plt.title(f'Parameters:\n{param_str}')
                plt.xticks(rotation=45)
                plt.ylabel('Normalized Order Difference')

        plt.tight_layout()
        plt.savefig(plot_dir / f'{self.domain}_order_differences.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _save_raw_data(self, results, file_dir):
        data_filename = file_dir / f'heuristic_comparision_{self.domain}_results.csv'

        # Combine all results into one DataFrame with parameter information
        all_data = []
        for params, df in results:
            df_with_params = df.copy()
            for param_name, param_value in params.items():
                df_with_params[param_name] = param_value
            all_data.append(df_with_params)

        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df.to_csv(data_filename, index=False)

def main():
    """
    Main function to run heuristic comparisons for different problems using parallel processing.
    This function performs the following steps:
    1. Determines the number of CPU cores to use for parallel processing,
    defaulting to 75% of available cores.
    2. Logs the number of workers being used.
    3. Runs a parameter study for the Sliding Puzzle problem using the specified number of workers.
    4. Logs the completion of the Sliding Puzzle parameter study.
    5. Runs a parameter study for the Blocks World problem using the specified number of workers.
    6. Logs the completion of the Blocks World parameter study.
    """

    # Use 75% of available CPU cores by default
    max_workers = max(1, int(mp.cpu_count() * 0.75))
    logger.info("Using up to %d workers for parallel processing", max_workers)

    logger.info("Running Sliding Puzzle parameter study...")
    sp_comparison = HeuristicComparison('sliding_puzzle', max_workers=max_workers)
    sp_comparison.run_parameter_study()

    logger.info("Running Blocks World parameter study...")
    bw_comparison = HeuristicComparison('blocks_world', max_workers=max_workers)
    bw_comparison.run_parameter_study()

if __name__ == "__main__":
    main()
