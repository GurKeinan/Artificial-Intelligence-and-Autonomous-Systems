import pickle
import os
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from sliding_puzzle_A_star import SearchNode


def load_data(root_dir):
    data_list = []
    # Iterate through the pickle files
    for file_name in os.listdir(root_dir):
        if file_name.endswith('.pkl'):
            with open(os.path.join(root_dir, file_name), 'rb') as f:
                tree = pickle.load(f)
                data_list.append(tree)
    return data_list


def rmse(predictions, targets):
    return ((predictions - targets) ** 2).mean() ** 0.5


def vesp_benchmark(root):
    nodes = []
    targets = []

    def traverse(node, parent_id=None):
        node_id = len(targets)

        # Extract vesp
        if node.serial_number == 0:
            nodes.append(0)
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

    rmse = rmse(nodes, targets)
    print(f"RMSE: {rmse}")
    return nodes, targets, rmse


def vasp_benchmark(root, window_size=10):
    e_vals = []
    nodes = []
    targets = []

    def traverse(node, parent_id=None):
        node_id = len(targets)

        # Extract vesp
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

    rmse = rmse(nodes, targets)
    print(f"RMSE: {rmse}")
    return nodes, targets, rmse


def pbp_benchmark(root):
    nodes = []
    targets = []

    def traverse(node, parent_id=None):
        node_id = len(targets)

        # Extract PBP
        pbp = node.g / (node.h + node.g)
        nodes.append(pbp)

        targets.append(node.progress)

        # Recursively traverse children
        for child in node.children:
            traverse(child, parent_id=node_id)

    # Start traversal from the root
    traverse(root)

    rmse = rmse(nodes, targets)
    print(f"RMSE: {rmse}")
    return nodes, targets, rmse


def random_forest_benchmark(root):

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
    X_train, X_test, y_train, y_test = train_test_split(
        nodes, targets, test_size=0.2, random_state=42)
    regr = RandomForestRegressor(max_depth=3, random_state=0)
    regr.fit(X_train, y_train)

    # Predict on test set:
    y_pred = regr.predict(X_test)
    rmse = rmse(y_test, y_pred)
    print(f"RMSE on test: {rmse}")

    # predict on all nodes:
    y_pred_full = regr.predict(nodes)
    rmse = rmse(targets, y_pred)
    print(f"RMSE on all nodes: {rmse}")

    return y_pred_full, targets, rmse


def main():
    root_dir = "code/puzzle_tree_dataset"
    data = load_data(root_dir)
    print(f"Loaded {len(data)} search trees.")

    results = []
    for tree in data:
        nodes, targets, rmse = vesp_benchmark(tree)
        results.append((nodes, targets, rmse))

    # Calculate average RMSE
    print(f"Average RMSE: {sum([r[2] for r in results]) / len(results)}")


if __name__ == "__main__":
    main()
