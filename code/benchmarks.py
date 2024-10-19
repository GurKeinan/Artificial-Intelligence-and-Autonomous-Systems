import pickle
import os
from pathlib import Path
import random

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

from sliding_puzzle_A_star import SearchNode


def load_data(root_dir):
    data_list = []
    name_list = []
    root_path = Path(root_dir) # all folders in the root directory

    datasets = '\n'.join([str(p.as_posix()) for p in root_path.iterdir() if p.name != '.DS_Store'])
    print(f"Read Datasets:\n{datasets}")

    for pkl_file in root_path.rglob('*.pkl'):
        with pkl_file.open('rb') as f:
            tree = pickle.load(f)
            name_list.append(root_path / pkl_file)
            data_list.append(tree)

    return data_list, name_list


def vesp_benchmark(root, print_rmse=False):
    nodes = []
    targets = []

    def traverse(node, parent_id=None):
        # print all node features:
        # print(f"Serial: {node.serial_number}, Parent Serial: {node.parent.serial_number if node.parent != None else None}, g: {node.g}, h: {node.h}, f: {node.f}, child_count: {node.child_count}, h_0: {node.h_0}, min_h_seen: {node.min_h_seen}, nodes_since_min_h: {node.nodes_since_min_h}, max_f_seen: {node.max_f_seen}, nodes_since_max_f: {node.nodes_since_max_f}\n")
        # print(node.state)
        # print("\n")

        node_id = len(targets)

        # Extract vesp
        if node.serial_number == 0 or node.h_0 == node.min_h_seen:
            vesp = 0
        else:
            v = (node.h_0 - node.min_h_seen) / node.serial_number
            se_v = node.min_h_seen / v
            vesp = node.serial_number / (node.serial_number + se_v)
        
        nodes.append(vesp)
        targets.append(node.progress)

        # Recursively traverse children
        for child in node.children:
            traverse(child, parent_id=node_id)

    # Start traversal from the root
    traverse(root)

    rmse = mean_squared_error(targets, nodes, squared=False)
    if print_rmse:
        print(f"RMSE: {rmse}")
    
    return nodes, targets, rmse


def vasp_benchmark(root, window_size=10, print_rmse=False):
    e_vals = []
    nodes = []
    targets = []

    def traverse(node, parent_id=None):
        node_id = len(targets)

        # Extract vasp
        if node.serial_number == 0:
            vasp = 0
        else:
            e_vals.append(node.serial_number - node.parent.serial_number)
            if len(e_vals) < window_size:
                window_average = sum(e_vals) / len(e_vals)
            else:
                window_average = sum(e_vals[-window_size:]) / \
                    len(e_vals[-window_size:])

            se_e = window_average * node.min_h_seen
            vasp = node.serial_number / (node.serial_number + se_e)
        nodes.append(vasp)

        targets.append(node.progress)

        # Recursively traverse children
        for child in node.children:
            traverse(child, parent_id=node_id)

    # Start traversal from the root
    traverse(root)

    rmse = mean_squared_error(targets, nodes, squared=False)
    if print_rmse:
        print(f"RMSE: {rmse}")
    
    return nodes, targets, rmse


def pbp_benchmark(root, print_rmse=False):
    nodes = []
    targets = []

    def traverse(node, parent_id=None):
        node_id = len(targets)

        # Extract PBP
        if node.g == 0 and node.h == 0:
            pbp = 0 #TODO: maybe 1?
        else:
            pbp = node.g / (node.h + node.g)
        nodes.append(pbp)

        targets.append(node.progress)

        # Recursively traverse children
        for child in node.children:
            traverse(child, parent_id=node_id)

    # Start traversal from the root
    traverse(root)

    rmse = mean_squared_error(targets, nodes, squared=False)
    if print_rmse:
        print(f"RMSE: {rmse}")

    return nodes, targets, rmse


def random_forest_benchmark(root, print_rmse=False):
    nodes = []
    targets = []

    def traverse(node, parent_id=None):
        node_id = len(nodes)
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
        nodes.append(node_features)
        targets.append(node.progress)  # Using 'f' as the regression target

        # Recursively traverse children
        for child in node.children:
            traverse(child, node_id)

    # Start traversal from the root
    traverse(root)

    # Random Forest Regression:
    regr = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42) 
    regr.fit(nodes, targets)

    # Predict on all nodes:
    y_pred_full = regr.predict(nodes)
    rmse_full = mean_squared_error(targets, y_pred_full, squared=False)

    if print_rmse:
        print(f"RMSE: {rmse_full}")

    return y_pred_full, targets, rmse_full


def main():

    ### Load data
    base_dir = Path(__file__).resolve().parent
    if base_dir.name != "code":
        base_dir = base_dir / "code"
    data_dir = base_dir / "dataset"

    data, names = load_data(data_dir)
    print(f"Loaded {len(data)} search trees.")

    ### Benchmarks
    benchmark_models = [vesp_benchmark, vasp_benchmark, pbp_benchmark, random_forest_benchmark]

    for benchmark_model in benchmark_models:
        results = []
        for tree, name in zip(data, names):
            nodes, targets, rmse = benchmark_model(tree)
            results.append((nodes, targets, rmse))

        # Calculate average RMSE
        print(f"Average RMSE for {benchmark_model}: {sum([r[2] for r in results]) / len(results)}")


if __name__ == "__main__":
    main()
