from datetime import datetime
import pickle
import os
from pathlib import Path
import random
import logging

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from tqdm import tqdm

from general_state import StateInterface, SearchNode

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Create timestamp for the log file
timestamp = datetime.now().strftime('%d.%m.%Y_%H:%M:%S')
log_filename = log_dir / f"benchmarks_{timestamp}.log"

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
    logger.debug(f"Tree properties - Nodes: {num_nodes}")
    return (num_nodes <= max_nodes)

def load_filtered_data(root_dir, max_nodes=10000):
    data_list = []
    name_list = []
    root_path = Path(root_dir)

    accepted_count = 0
    rejected_count = 0

    datasets = '\n'.join([str(p.as_posix()) for p in root_path.iterdir() if p.name != '.DS_Store'])
    logger.info(f"Read Datasets:\n{datasets}")

    total_files = sum(1 for _ in root_path.rglob('*.pkl'))
    logger.info(f"Found {total_files} PKL files")

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
        except Exception as e:
            logger.warning(f"Failed to process {pkl_file}: {str(e)}")
            continue

    logger.info(f"Processing Summary:")
    logger.info(f"- Accepted trees: {accepted_count}")
    logger.info(f"- Rejected trees (too complex): {rejected_count}")

    return data_list, name_list

def compute_score(nodes, targets, print_res):
    nodes = np.array(nodes)
    targets = np.array(targets)
    sse = sum((nodes - targets) ** 2)
    return sse

def vesp_benchmark(root, print_res=False):
    nodes = []
    targets = []

    def traverse(node, parent_id=None):
        node_id = len(targets)

        if node.serial_number == 0 or node.h_0 == node.min_h_seen:
            vesp = 0
        else:
            v = (node.h_0 - node.min_h_seen) / node.serial_number
            se_v = node.min_h_seen / v
            vesp = node.serial_number / (node.serial_number + se_v)

        nodes.append(vesp)
        targets.append(node.progress)

        for child in node.children:
            traverse(child, parent_id=node_id)

    traverse(root)
    res = compute_score(nodes, targets, print_res)
    return nodes, targets, res

def vasp_benchmark(root, window_size=50, print_res=False):
    e_vals = []
    nodes = []
    targets = []

    def traverse(node, parent_id=None):
        node_id = len(targets)

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
            traverse(child, parent_id=node_id)

    traverse(root)
    res = compute_score(nodes, targets, print_res)
    return nodes, targets, res

def pbp_benchmark(root, print_res=False):
    nodes = []
    targets = []

    def traverse(node, parent_id=None):
        node_id = len(targets)

        if node.g == 0 and node.h == 0:
            pbp = 0
        else:
            pbp = node.g / (node.h + node.g)
        nodes.append(pbp)
        targets.append(node.progress)

        for child in node.children:
            traverse(child, parent_id=node_id)

    traverse(root)
    res = compute_score(nodes, targets, print_res)
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

def random_forest_benchmark(trees, print_res=False):
    """Train and evaluate random forest on all trees combined."""
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

    logger.info(f"Total nodes collected: {len(all_features)}")

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        all_features, all_targets, test_size=0.2, random_state=42
    )

    # Train model
    logger.info("Training Random Forest model...")
    regr = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    regr.fit(X_train, y_train)

    # Evaluate
    train_pred = regr.predict(X_train)
    test_pred = regr.predict(X_test)

    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)

    return regr, train_mse, test_mse

def plot_feature_importance(model, feature_names):
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
    base_dir = Path(__file__).resolve().parent
    if base_dir.name != "code":
        base_dir = base_dir / "code"
    data_dir = base_dir / "dataset"

    # Load filtered data
    data, names = load_filtered_data(
        root_dir=data_dir,
        max_nodes=20000
    )
    logger.info(f"Loaded {len(data)} filtered search trees.")

    # Traditional benchmarks
    benchmark_models = [vesp_benchmark, vasp_benchmark, pbp_benchmark]
    total_samples = 0

    for benchmark_model in benchmark_models:
        results = []
        logger.info(f"Running {benchmark_model.__name__}...")

        for tree, name in tqdm(zip(data, names), total=len(data)):
            try:
                nodes, targets, sse = benchmark_model(tree, print_res=False)
                results.append((nodes, targets, sse))
                total_samples += len(nodes)
            except Exception as e:
                logger.warning(f"Failed to process tree {name} with {benchmark_model.__name__}: {str(e)}")
                continue

        if results:
            mse = sum([r[2] for r in results]) / total_samples
            logger.info(f"MSE for {benchmark_model.__name__}: {mse}")
        else:
            logger.warning(f"No successful results for {benchmark_model.__name__}")

    # Random Forest benchmark (on all trees combined)
    logger.info("\nRunning Random Forest benchmark...")
    feature_names = [
        'serial_number', 'g', 'h', 'f', 'child_count',
        'h_0', 'min_h_seen', 'nodes_since_min_h',
        'max_f_seen', 'nodes_since_max_f'
    ]

    rf_model, train_mse, test_mse = random_forest_benchmark(data)
    logger.info(f"Train MSE for Random Forest: {train_mse:.4f}")
    logger.info(f"Test MSE for Random Forest: {test_mse:.4f}")
    plot_feature_importance(rf_model, feature_names)

if __name__ == "__main__":
    main()