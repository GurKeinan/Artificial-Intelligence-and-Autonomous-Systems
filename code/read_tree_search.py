import pickle
import os
from pathlib import Path
import random

import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader

from sliding_puzzle_A_star import SearchNode

class TreeDataset(InMemoryDataset):
    def __init__(self, root_dir, test_ratio=0, transform=None, pre_transform=None):
        super(TreeDataset, self).__init__(
            root_dir, transform, pre_transform)
        self.root_dir = root_dir
        self.test_ratio = test_ratio
        self.data, self.slices = self.load_data()

    # def load_data(self):
    #     data_list = []
    #     # Iterate through the pickle files
    #     for file_name in os.listdir(self.root_dir):
    #         if file_name.endswith('.pkl'):
    #             with open(os.path.join(self.root_dir, file_name), 'rb') as f:
    #                 tree = pickle.load(f)
    #                 graph_data = tree_to_graph(tree, self.test_ratio)
    #                 data_list.append(graph_data)
    #     return self.collate(data_list)  # Collate into InMemoryDataset format
    
    def load_data(self):
        data_list = []
        name_list = []
        root_path = Path(self.root_dir) # all folders in the root directory

        datasets = '\n'.join([str(p.as_posix()) for p in root_path.iterdir() if p.name != '.DS_Store'])
        print(f"Read Datasets:\n{datasets}")

        for pkl_file in root_path.rglob('*.pkl'):
            with pkl_file.open('rb') as f:
                tree = pickle.load(f)
                name_list.append(root_path / pkl_file)

                graph_data = tree_to_graph(tree, self.test_ratio)
                data_list.append(graph_data)

        return self.collate(data_list)


def tree_to_graph(root, test_ratio=0):
    """ Converts a binary search tree to a PyTorch Geometric graph."""
    if root is None:
        return None

    nodes = []
    edges = []
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

        # If there is a parent node, add an edge
        if parent_id is not None:
            edges.append([parent_id, node_id])

        # Recursively traverse children
        for child in node.children:
            traverse(child, node_id)


    # Start traversal from the root
    traverse(root)

    # Convert data to torch tensors
    x = torch.tensor(nodes, dtype=torch.float)  # Node features (f, g, h)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # Edges
    y = torch.tensor(targets, dtype=torch.float)  # Regression targets

    if (test_ratio > 0):
        num_nodes = len(nodes)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        # Randomly assign nodes to training or testing set
        train_indices = random.sample(
            range(num_nodes), int((1-test_ratio) * num_nodes))
        train_mask[train_indices] = True
        test_mask[~train_mask] = True

        return Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)

    else:
        return Data(x=x, edge_index=edge_index, y=y)


def main():
    data_path = "code/puzzle_tree_dataset/"

    # Load the dataset:
    dataset = TreeDataset(root_dir=data_path, test_ratio=0.2)
    print(f"Dataset size: {len(dataset)}")

    # Create dataloaders:
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    print(f"Loader batches: {len(loader)}")
    

if __name__ == '__main__':
    main()