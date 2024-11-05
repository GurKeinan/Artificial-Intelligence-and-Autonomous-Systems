"""
This script performs various benchmarks on search trees,
including traditional benchmarks (vesp_benchmark, vasp_benchmark, pbp_benchmark)
and a Random Forest benchmark.
It logs the results and plots feature importance for the Random Forest model.

Functions:
    - analyze_tree(root): Analyzes a search tree for its properties.
    - is_tree_acceptable(root, max_nodes=10000): Checks if a tree meets the criteria for inclusion.
    - load_filtered_data(root_dir, max_nodes): Loads and filters data from a specified directory.
    - compute_score(nodes, targets): Computes the sum of squared errors (SSE) between two arrays.
    - vesp_benchmark(root): Recursively traverses a tree structure,
    calculating vesp score and appending it to the nodes list.
    - vasp_benchmark(root, window_size=50): Performs a VASP benchmark on a tree structure.
    - pbp_benchmark(root): Performs a PBP (Progress-Based Planning) benchmark on a tree structure.
    - collect_tree_data(root): Collects features and targets from a single tree.
    - random_forest_benchmark(trees): Benchmarks a Random Forest model.
    - plot_feature_importance(model, feature_names): Plots the feature importance of a given model.
    - main(): Main function to run benchmarks on filtered search trees.

Usage:
    Run this script directly to execute the main function,
    which performs the benchmarks and logs the results.
"""
from datetime import datetime
import pickle
from pathlib import Path
import logging
import sys
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from tqdm import tqdm
from time import time

repo_root = Path(__file__).resolve().parent
dataset_creation_path = repo_root / "dataset_creation"
sys.path.append(str(dataset_creation_path))

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Create timestamp for the log file
log_filename = log_dir / "benchmarks_with_runtime.log"
if log_filename.exists():
    log_filename.unlink()

# Create file handler with immediate flush
file_handler = logging.FileHandler(log_filename)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

def analyze_tree(root):
    """Analyze a search tree for its properties."""
    num_nodes = 0

    def traverse(node, depth=0):
        nonlocal num_nodes
        num_nodes += 1

        for child in node.children:
            traverse(child, depth + 1)

    traverse(root)
    return num_nodes

def is_tree_acceptable(root, max_nodes=10000):
    """Check if a tree meets our criteria for inclusion."""
    num_nodes = analyze_tree(root)
    return num_nodes <= max_nodes

def load_filtered_data(root_dir, max_nodes):
    """
    Load and filter data from a specified directory based on the complexity of the data structures.
    This function reads all `.pkl` files from the given root directory,
    deserializes them, and filters
    them based on the number of nodes in the data structure.
    Only data structures with a number of nodes
    less than or equal to `max_nodes` are accepted.
    Args:
        root_dir (str or Path): The root directory containing the `.pkl` files.
        max_nodes (int): The maximum number of nodes allowed in the data structure.
    Returns:
        tuple: A tuple containing two lists:
            - data_list (list): A list of accepted data structures.
            - name_list (list): A list of paths corresponding to the accepted data structures.
    Raises:
        Exception: If there is an error in processing a `.pkl` file,
        it logs a warning and continues with the next file.
    Logs:
        - The datasets read from the directory.
        - The total number of `.pkl` files found.
        - A summary of the processing, including the number of accepted and rejected search trees.
    """

    data_list = []
    name_list = []
    root_path = Path(root_dir)

    accepted_count = 0
    rejected_count = 0

    datasets = '\n'.join([str(p.as_posix()) for p in root_path.iterdir() if p.name != '.DS_Store'])
    logger.info("Read Datasets: \n %s", datasets)

    total_files = sum(1 for _ in root_path.rglob('*.pkl'))
    logger.info("Found %d PKL files in the dataset.", total_files)

    for pkl_file in tqdm(root_path.rglob('*.pkl'), total=total_files):
        try:
            with pkl_file.open('rb') as f:
                tree = pickle.load(f)
                if is_tree_acceptable(tree, max_nodes):
                    name_list.append(root_path / pkl_file)
                    data_list.append(tree)
                    accepted_count += 1
                else:
                    rejected_count += 1
        except Exception as e: # pylint: disable=broad-except
            logger.warning("Failed to process %s: %s", pkl_file, str(e))
            continue

    logger.info("Processing Summary:")
    logger.info("- Accepted trees: %d", accepted_count)
    logger.info("- Rejected trees (too complex): %d", rejected_count)

    return data_list, name_list

def compute_score(nodes, targets):
    """
    Compute the sum of squared errors (SSE) between two arrays.
    Parameters:
    nodes (list or array-like): The first array of values.
    targets (list or array-like): The second array of values to compare against.
    Returns:
    float: The sum of squared errors between the nodes and targets.
    """

    nodes = np.array(nodes)
    targets = np.array(targets)
    sse = sum((nodes - targets) ** 2)
    return sse

def vesp_benchmark(root,):

    """
        Recursively traverses a tree structure starting from the given node,
        calculating a vesp value and appending it to the nodes list.
        Args:
            node (Node): The starting node for the traversal. Must have the following attributes:
                - serial_number (int): A unique identifier for the node.
                - h_0 (float): An initial heuristic value associated with the node.
                - min_h_seen (float): The minimum heuristic value seen so far in the traversal.
                - progress (Any): The progress value associated with the node.
                - children (list): A list of child nodes to be traversed.
        Side Effects:
            - Appends calculated vesp values to the global list `nodes`.
            - Appends node progress values to the global list `targets`.
    """

    nodes = []
    targets = []

    def traverse(node):
        """
        Traverse a tree structure starting from the given node, calculating and
        appending values to the nodes and targets lists.
        The function calculates a value `vesp` based on the node's serial number,
        initial heuristic value (h_0), and the minimum heuristic value seen (min_h_seen).
        It then appends `vesp` to the `nodes` list and the node's progress to the
        `targets` list. The function recursively traverses all children of the node.
        Args:
            node (Node): The starting node for the traversal. The node is expected
                         to have the following attributes:
                         - serial_number (int): The serial number of the node.
                         - h_0 (float): The initial heuristic value of the node.
                         - min_h_seen (float): The minimum heuristic value seen.
                         - progress (Any): The progress value of the node.
                         - children (list): A list of child nodes.
        Returns:
            None
        """

        if node.serial_number == 0 or node.h_0 == node.min_h_seen:
            vesp = 0
        else:
            v = (node.h_0 - node.min_h_seen) / node.serial_number
            se_v = node.min_h_seen / v
            vesp = node.serial_number / (node.serial_number + se_v)

        nodes.append(vesp)
        targets.append(node.progress)

        for child in node.children:
            traverse(child)

    traverse(root)
    res = compute_score(nodes, targets)
    return nodes, targets, res

def vasp_benchmark(root, window_size=50):
    """
    Perform a VASP (Value of Average Subtree Progress) benchmark on a tree structure.
    This function traverses a tree starting from the root node and calculates the VASP value
    for each node based on the serial numbers of the nodes and their parents. It uses a
    sliding window to compute the average of the differences in serial numbers.
    Args:
        root (Node): The root node of the tree to be traversed.
        window_size (int, optional): The size of the sliding window for averaging. Defaults to 50.
    Returns:
        tuple: A tuple containing three elements:
            - nodes (list): A list of VASP values for each node.
            - targets (list): A list of progress values for each node.
            - res: The result of the compute_score function applied to the nodes and targets.
    """

    e_vals = []
    nodes = []
    targets = []

    def traverse(node):

        if node.serial_number == 0:
            vasp = 0
        else:
            e_vals.append(node.serial_number - node.parent.serial_number)
            if len(e_vals) < window_size:
                window_average = sum(e_vals) / len(e_vals)
            else:
                window_average = sum(e_vals[-window_size:]) / len(e_vals[-window_size:])

            se_e = window_average * node.min_h_seen
            vasp = node.serial_number / (node.serial_number + se_e)
        nodes.append(vasp)
        targets.append(node.progress)

        for child in node.children:
            traverse(child)

    traverse(root)
    res = compute_score(nodes, targets)
    return nodes, targets, res

def pbp_benchmark(root):
    """
    Perform a PBP (Progress-Based Planning) benchmark on a tree structure.
    This function traverses a tree starting from the root node, calculates the
    PBP value for each node, and collects the progress values. It then computes
    a score based on the collected PBP values and progress values.
    Args:
        root (Node): The root node of the tree to be traversed.
    Returns:
        tuple: A tuple containing three elements:
            - nodes (list of float): The list of PBP values for each node.
            - targets (list of float): The list of progress values for each node.
            - res (float): The computed score based on the nodes and targets.
    """

    nodes = []
    targets = []

    def traverse(node):

        if node.g == 0 and node.h == 0:
            pbp = 0
        else:
            pbp = node.g / (node.h + node.g)
        nodes.append(pbp)
        targets.append(node.progress)

        for child in node.children:
            traverse(child)

    traverse(root)
    res = compute_score(nodes, targets)
    return nodes, targets, res

def collect_tree_data(root):
    """Collect features and targets from a single tree."""
    features = []
    targets = []

    def traverse(node):
        node_features = [
            node.serial_number,
            node.g,
            node.h,
            node.f,
            node.child_count,
            node.h_0,
            node.min_h_seen,
            node.nodes_since_min_h,
            node.max_f_seen,
            node.nodes_since_max_f,
        ]
        features.append(node_features)
        targets.append(node.progress)

        for child in node.children:
            traverse(child)

    traverse(root)
    return features, targets

def random_forest_benchmark(trees):
    """
    Benchmarks a Random Forest model using data collected from a list of decision trees.
    Parameters:
    trees (list): A list of decision tree objects from which to collect data.
    Returns:
    tuple: A tuple containing the trained Random Forest model,
    the mean squared error on the training set,
    and the mean squared error on the test set.
    """

    logger.info("Collecting data from all trees...")
    all_features = []
    all_targets = []

    # Collect data from all trees
    for tree in tqdm(trees):
        features, targets = collect_tree_data(tree)
        all_features.extend(features)
        all_targets.extend(targets)

    all_features = np.array(all_features)
    all_targets = np.array(all_targets)

    logger.info("Total nodes collected: %d", len(all_features))

    # Split data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        all_features, all_targets, test_size=0.2, random_state=42
    )

    # Train model
    logger.info("Training Random Forest model...")
    regr = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    regr.fit(x_train, y_train)

    # Evaluate
    start_time = time()
    train_pred = regr.predict(x_train)
    test_pred = regr.predict(x_test)

    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)

    logger.info("Time taken for Random Forest: %.2f seconds", time() - start_time)

    return regr, train_mse, test_mse

def plot_feature_importance(model, feature_names):
    """
    Plots the feature importance of a given model.
    Parameters:
        model (object): The trained model with feature importances.
        It should have an attribute `feature_importances_`.
        feature_names (list of str): List of feature names used in the model.
    Returns:
        None
    """

    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance in Random Forest Model")
    plt.bar(range(len(importance)), importance[indices])
    plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)
    plt.ylabel('Feature Importance')
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to run benchmarks on filtered search trees.
    This function performs the following steps:
    1. Sets up the base directory and data directory paths.
    2. Loads filtered data from the dataset directory.
    3. Runs traditional benchmarks (vesp_benchmark, vasp_benchmark, pbp_benchmark) on the data.
    4. Logs the Mean Squared Error (MSE) for each benchmark model.
    5. Runs a Random Forest benchmark on all combined trees.
    6. Logs the train and test MSE for the Random Forest model.
    7. Plots the feature importance for the Random Forest model.
    The function handles exceptions during the benchmark runs and logs warnings for any failures.
    Raises:
        Exception: If any error occurs during the processing of a tree with a benchmark model.
    """

    base_dir = Path(__file__).resolve().parent
    if base_dir.name != "code":
        base_dir = base_dir / "code"
    data_dir = base_dir / "dataset"

    # Load filtered data
    data, names = load_filtered_data(
        root_dir=data_dir,
        max_nodes=15000
    )
    logger.info("Loaded %d filtered search trees.", len(data))

    # Traditional benchmarks
    benchmark_models = [vesp_benchmark, vasp_benchmark, pbp_benchmark]
    total_samples = 0

    for benchmark_model in benchmark_models:
        results = []
        logger.info("Running %s...", benchmark_model.__name__)
        start_time = time()

        for tree, name in tqdm(zip(data, names), total=len(data)):
            try:
                nodes, targets, sse = benchmark_model(tree)
                results.append((nodes, targets, sse))
                total_samples += len(nodes)
            except Exception as e: # pylint: disable=broad-except
                logger.warning("Failed to process tree %s with %s: %s",
                               name, benchmark_model.__name__, str(e))
                continue

        if results:
            mse = sum([r[2] for r in results]) / total_samples
            logger.info("MSE for %s: %.4f", benchmark_model.__name__, mse)
        else:
            logger.warning("No successful results for %s.", benchmark_model.__name__)
        
        logger.info("Time taken for %s: %.2f seconds", benchmark_model.__name__, time() - start_time)

    # Random Forest benchmark (on all trees combined)
    logger.info("Running Random Forest benchmark...")
    feature_names = [
        'serial_number', 'g', 'h', 'f', 'child_count',
        'h_0', 'min_h_seen', 'nodes_since_min_h',
        'max_f_seen', 'nodes_since_max_f'
    ]

    rf_model, train_mse, test_mse = random_forest_benchmark(data)
    logger.info("Train MSE for Random Forest: %.4f", train_mse)
    logger.info("Test MSE for Random Forest: %.4f", test_mse)
    plot_feature_importance(rf_model, feature_names)

if __name__ == "__main__":
    main()
