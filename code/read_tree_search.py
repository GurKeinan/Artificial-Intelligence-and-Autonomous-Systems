import pickle
import os
from pathlib import Path
import random
from tqdm import tqdm
import logging

import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader

from general_state import StateInterface, SearchNode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SerializableDataLoader:
    """Wrapper class to make DataLoader serializable"""
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def __iter__(self):
        return iter(self._loader)

    def __len__(self):
        return len(self._loader)

class TreeDataset(InMemoryDataset):
    def __init__(self, root_dir, test_ratio=0, transform=None, pre_transform=None):
        super(TreeDataset, self).__init__(
            root_dir, transform, pre_transform)
        self.root_dir = root_dir
        self.test_ratio = test_ratio
        self.data, self.slices = self.load_data()

    def load_data(self):
        data_list = []
        name_list = []
        root_path = Path(self.root_dir) # all folders in the root directory

        datasets = '\n'.join([str(p.as_posix()) for p in root_path.iterdir() if p.name != '.DS_Store'])
        logger.info(f"Read Datasets:\n{datasets}")

        for pkl_file in tqdm(root_path.rglob('*.pkl')):
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

def save_processed_data(loader, save_path):
    """Save a processed DataLoader to disk."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract dataset from loader
    dataset = loader.dataset

    # Create save dictionary with all necessary information
    save_dict = {
        'data_list': [data for data in dataset],
        'batch_size': loader.batch_size,
        'shuffle': loader.shuffle,
    }

    try:
        torch.save(save_dict, save_path)
        logger.info(f"Successfully saved processed data to {save_path}")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise

def load_processed_data(load_path):
    """Load a processed DataLoader from disk."""
    load_path = Path(load_path)

    if not load_path.exists():
        raise FileNotFoundError(f"No saved data found at {load_path}")

    try:
        save_dict = torch.load(load_path)

        # Reconstruct dataset
        dataset = InMemoryDataset()
        dataset._data_list = save_dict['data_list']

        # Create new loader
        loader = SerializableDataLoader(
            dataset,
            batch_size=save_dict['batch_size'],
            shuffle=save_dict['shuffle']
        )

        logger.info(f"Successfully loaded processed data from {load_path}")
        return loader

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def get_dataloaders(root_dir, processed_path=None, batch_size=32, test_ratio=0.2):
    """Get DataLoader either from processed cache or create new one."""
    if processed_path:
        processed_path = Path(processed_path)

        if processed_path.exists():
            logger.info("Found processed data, loading from cache...")
            return load_processed_data(processed_path)

    logger.info("Creating new DataLoader...")
    dataset = TreeDataset(root_dir=root_dir, test_ratio=test_ratio)
    loader = SerializableDataLoader(dataset, batch_size=batch_size, shuffle=True)

    if processed_path:
        # Save for future use
        save_processed_data(loader, processed_path)

    return loader

def main():
    base_dir = Path(__file__).resolve().parent
    if base_dir.name != "code":
        base_dir = base_dir / "code"

    data_dir = base_dir / "dataset"
    processed_path = base_dir / "processed" / "dataloader.pt"

    # This will either load cached data or create new
    loader = get_dataloaders(
        data_dir,
        processed_path=processed_path,
        batch_size=4,
        test_ratio=0.2
    )

    print(f"Dataset loaded with {len(loader)} batches")

if __name__ == '__main__':
    main()