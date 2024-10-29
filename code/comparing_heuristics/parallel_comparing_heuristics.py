import sys
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, Set, List, Tuple
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

repo_root = Path(__file__).resolve().parent.parent
dataset_creation_path = repo_root / "dataset_creation"
sys.path.append(str(dataset_creation_path))

from sliding_puzzle_generator import generate_sliding_puzzle_problem
from block_world_generator import generate_block_world_problem
from general_A_star import a_star
from sliding_puzzle_heuristics import sp_manhattan_distance, sp_misplaced_tiles, sp_h_max
from block_world_heuristics import bw_misplaced_blocks, bw_height_difference, bw_h_max

# Create logs directory if it doesn't exist
current_dir = Path(__file__).resolve().parent
log_dir = current_dir / "logs"
log_dir.mkdir(exist_ok=True)

def configure_logging():
    log_file = log_dir / "app.log"
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

configure_logging()
logger = logging.getLogger(__name__)

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
    def __init__(self, domain: str, max_workers: int = None):
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
            desc = f"Generating blocks world problems - num_blocks={kwargs['num_blocks']}, num_stacks={kwargs['num_stacks']}, num_moves={kwargs['num_moves']}"

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

            for future in tqdm(as_completed(futures), total=len(problems),
                             desc="Solving problems"):
                success, results = future.result()
                if success:
                    metrics = self._analyze_search_trees(results)
                    all_results.append(metrics)
        logger.info("Solved %d problems successfully", len(all_results))
        return all_results

    def _analyze_search_trees(self, problem_results: Dict) -> Dict:
        metrics = {}
        heuristics = list(problem_results.keys())

        # Basic metrics
        for heur in heuristics:
            tree = problem_results[heur]['search_tree']
            metrics[f'total_nodes_{heur}'] = self._count_nodes(tree)
            metrics[f'solution_length_{heur}'] = problem_results[heur]['solution_length']

        lengths = [problem_results[heur]['solution_length'] for heur in heuristics]
        metrics['solution_lengths_identical'] = len(set(lengths)) == 1

        for i, heur1 in enumerate(heuristics):
            for heur2 in heuristics[i+1:]:
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
                        order_diffs = []
                        for state in shared_states:
                            norm_serial1 = map1[state] / max_serial1
                            norm_serial2 = map2[state] / max_serial2
                            order_diff = abs(norm_serial1 - norm_serial2)
                            order_diffs.append(order_diff)

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

    def run_comparison(self, num_problems: int, save_results=True, **kwargs):
        """Run full comparison and save results"""
        problems = self.generate_problems(num_problems, **kwargs)
        results = self.solve_with_heuristics(problems)

        if not results:
            print("No problems were solved successfully by all heuristics")
            return None

        df = pd.DataFrame(results)

        if save_results:
            filename = f"heuristic_comparison_{self.domain}.csv"
            df.to_csv(filename, index=False)

        return df

    def run_parameter_study(self):
        """Run parameter study in parallel"""
        if self.domain == 'sliding_puzzle':
            params = [
                {'size': 4, 'num_moves': 5},
                {'size': 4, 'num_moves': 8},
                {'size': 4, 'num_moves': 11},
                {'size': 6, 'num_moves': 5},
                {'size': 6, 'num_moves': 10},
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

        # Plot node counts
        n_cols = min(3, n_comparisons)
        n_rows = (n_comparisons + n_cols - 1) // n_cols

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

        # Plot state space overlap
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

            if idx == 0:  # Only show legend on first subplot
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(plot_dir / f'{self.domain}_state_overlap.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Plot solution lengths if differences exist
        if any(not df['solution_lengths_identical'].all() for _, df in results):
            plt.figure(figsize=(6*n_cols, 5*n_rows))
            plt.suptitle(f'{self.domain.title()} - Solution Lengths Across Heuristics', y=1.02)

            for idx, (params, df) in enumerate(results):
                plt.subplot(n_rows, n_cols, idx+1)
                length_cols = [col for col in df.columns if col.startswith('solution_length_')]
                df[length_cols].boxplot()

                non_identical = (~df['solution_lengths_identical']).sum()
                param_str = '\n'.join(f'{k}={v}' for k, v in params.items())

                if non_identical > 0:
                    plt.title(f'Parameters:\n{param_str}\n⚠️ {non_identical} cases with different lengths')
                else:
                    plt.title(f'Parameters:\n{param_str}\n(All solutions optimal)')

                plt.xticks(rotation=45)
                plt.ylabel('Solution Length')

            plt.tight_layout()
            plt.savefig(plot_dir / f'{self.domain}_solution_lengths.png', dpi=300, bbox_inches='tight')
            plt.close()

        # Plot order differences
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

        # Also save the raw data
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        data_filename = file_dir / f'{self.domain}_results_{timestamp}.csv'

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