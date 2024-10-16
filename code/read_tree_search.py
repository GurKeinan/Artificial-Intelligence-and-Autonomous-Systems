import pickle
import os
import numpy as np

import torch
from torch_geometric.data import InMemoryDataset, Data, DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from sliding_puzzle_generator import SlidingPuzzleState, generate_sliding_puzzle_problem
from sliding_puzzle_A_star import SearchNode


class TreeDataset(InMemoryDataset):
    def __init__(self, root_dir, transform=None, pre_transform=None):
        super(TreeDataset, self).__init__(root_dir, transform, pre_transform)
        self.root_dir = root_dir
        self.data, self.slices = self.load_data()

    def load_data(self):
        data_list = []
        # Iterate through the pickle files
        for file_name in os.listdir(self.root_dir):
            if file_name.endswith('.pickle'):
                with open(os.path.join(self.root_dir, file_name), 'rb') as f:
                    tree = pickle.load(f)
                    graph_data = tree_to_graph(tree)
                    data_list.append(graph_data)
        return self.collate(data_list)  # Collate into InMemoryDataset format

def read_from_file(file_path):
    with open(file_path, 'rb') as f:
        root = pickle.load(f)
    print(root)
    return root

def tree_to_graph(root):
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

    return Data(x=x, edge_index=edge_index, y=y)


def main():
    root = read_from_file('search_tree.pkl')

    # Example usage:
    graph_data = tree_to_graph(root)
    print(graph_data)

    # full dataset:
    data_path = 'puzzle_tree_dataset'
    dataset = TreeDataset(root_dir=data_path)

    # Create dataloaders:
    train_dataset = dataset[:int(len(dataset) * 0.8)]
    test_dataset = dataset[int(len(dataset) * 0.8):]

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    

if __name__ == '__main__':
    main()