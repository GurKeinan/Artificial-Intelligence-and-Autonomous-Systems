import pickle
import os
from pathlib import Path
import random

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

from general_state import StateInterface, SearchNode

def load_data(root_dir):
    data_list = []
    name_list = []
    root_path = Path(root_dir)

    datasets = '\n'.join([str(p.as_posix()) for p in root_path.iterdir() if p.name != '.DS_Store'])
    print(f"Read Datasets:\n{datasets}")

    for pkl_file in root_path.rglob('*.pkl'):
        with pkl_file.open('rb') as f:
            tree = pickle.load(f)
            name_list.append(root_path / pkl_file)
            data_list.append(tree)

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

def random_forest_benchmark(root, print_res=False):
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
        targets.append(node.progress)

        for child in node.children:
            traverse(child, node_id)

    traverse(root)

    if len(nodes) < 2:
        train_X, train_y = nodes, targets
    else:
        train_X, test_X, train_y, test_y = train_test_split(nodes, targets, test_size=0.2, random_state=42)
    regr = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42)
    regr.fit(train_X, train_y)

    y_pred_full = regr.predict(nodes)

    res = compute_score(y_pred_full, targets, print_res)
    return y_pred_full, targets, res, regr

def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance in Random Forest Model")
    plt.bar(range(len(importance)), importance[indices])
    plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()

def main():
    base_dir = Path(__file__).resolve().parent
    if base_dir.name != "code":
        base_dir = base_dir / "code"
    data_dir = base_dir / "dataset"

    data, names = load_data(data_dir)
    print(f"Loaded {len(data)} search trees.")

    benchmark_models = [vesp_benchmark, vasp_benchmark, pbp_benchmark, random_forest_benchmark]

    total_samples = 0
    for benchmark_model in benchmark_models:
        results = []
        for tree, name in zip(data, names):
            if benchmark_model.__name__ == 'random_forest_benchmark':
                nodes, targets, sse, model = benchmark_model(tree, print_res=False)
                feature_names = [
                    'serial_number', 'g', 'h', 'f', 'child_count',
                    'h_0', 'min_h_seen', 'nodes_since_min_h',
                    'max_f_seen', 'nodes_since_max_f'
                ]
                plot_feature_importance(model, feature_names)
            else:
                nodes, targets, sse = benchmark_model(tree, print_res=False)
            results.append((nodes, targets, sse))
            total_samples += len(nodes)

        mse = sum([r[2] for r in results]) / total_samples
        print(f"MSE for {benchmark_model.__name__}: {mse}")


if __name__ == "__main__":
    main()
