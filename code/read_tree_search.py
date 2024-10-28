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

# Set up logging
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
        self._loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        return iter(self._loader)

    def __len__(self):
        return len(self._loader)

class FilteredTreeDataset(InMemoryDataset):
    def __init__(self, root_dir, max_nodes=1000, max_depth=50, max_branching=20,
                 test_ratio=0, transform=None, pre_transform=None):
        self.max_nodes = max_nodes
        self.max_depth = max_depth
        self.max_branching = max_branching
        super(FilteredTreeDataset, self).__init__(root_dir, transform, pre_transform)
        self.root_dir = root_dir
        self.test_ratio = test_ratio
        try:
            self.data, self.slices = self.load_data()
        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            raise

    def analyze_tree(self, root):
        """Analyze a search tree for its properties."""
        num_nodes = 0
        max_depth = 0
        max_branch = 0

        def traverse(node, depth=0):
            nonlocal num_nodes, max_depth, max_branch
            num_nodes += 1
            max_depth = max(max_depth, depth)
            max_branch = max(max_branch, len(node.children))

            for child in node.children:
                traverse(child, depth + 1)

        traverse(root)
        return num_nodes, max_depth, max_branch

    def is_tree_acceptable(self, root):
        """Check if a tree meets our criteria for inclusion."""
        num_nodes, depth, branching = self.analyze_tree(root)

        # Log the tree properties for monitoring
        logger.debug(f"Tree properties - Nodes: {num_nodes}, Depth: {depth}, Max Branching: {branching}")

        return (num_nodes <= self.max_nodes and
                depth <= self.max_depth and
                branching <= self.max_branching)

    def load_data(self):
        data_list = []
        failed_files = []
        accepted_count = 0
        rejected_count = 0
        root_path = Path(self.root_dir)

        datasets = '\n'.join([str(p.as_posix()) for p in root_path.iterdir() if p.name != '.DS_Store'])
        logger.info(f"Read Datasets:\n{datasets}")

        total_files = sum(1 for _ in root_path.rglob('*.pkl'))
        logger.info(f"Found {total_files} PKL files")

        for pkl_file in tqdm(root_path.rglob('*.pkl'), total=total_files):
            try:
                if pkl_file.stat().st_size == 0:
                    logger.warning(f"Skipping empty file: {pkl_file}")
                    failed_files.append((pkl_file, "Empty file"))
                    continue

                with pkl_file.open('rb') as f:
                    try:
                        tree = pickle.load(f)
                    except (EOFError, pickle.UnpicklingError) as e:
                        logger.warning(f"Failed to load corrupted pickle file {pkl_file}: {str(e)}")
                        failed_files.append((pkl_file, f"Pickle error: {str(e)}"))
                        continue

                    if tree is None:
                        logger.warning(f"Skipping {pkl_file}: Tree is None")
                        failed_files.append((pkl_file, "Tree is None"))
                        continue

                    # Check if tree meets our criteria
                    if not self.is_tree_acceptable(tree):
                        rejected_count += 1
                        logger.debug(f"Rejected tree from {pkl_file} - too complex")
                        continue

                    try:
                        graph_data = tree_to_graph(tree, self.test_ratio)
                        if graph_data is not None:
                            data_list.append(graph_data)
                            accepted_count += 1
                        else:
                            logger.warning(f"Skipping {pkl_file}: tree_to_graph returned None")
                            failed_files.append((pkl_file, "Graph conversion failed"))
                    except Exception as e:
                        logger.warning(f"Failed to convert tree to graph for {pkl_file}: {str(e)}")
                        failed_files.append((pkl_file, f"Conversion error: {str(e)}"))
                        continue

            except Exception as e:
                logger.warning(f"Failed to process file {pkl_file}: {str(e)}")
                failed_files.append((pkl_file, f"Process error: {str(e)}"))
                continue

        # Log summary of processing
        logger.info(f"Processing Summary:")
        logger.info(f"- Accepted trees: {accepted_count}")
        logger.info(f"- Rejected trees (too complex): {rejected_count}")
        logger.info(f"- Failed processing: {len(failed_files)}")

        if failed_files:
            logger.warning("Failed to process the following files:")
            for file, reason in failed_files:
                logger.warning(f"  - {file}: {reason}")

        if not data_list:
            raise ValueError("No valid data was loaded from the dataset")

        try:
            return self.collate(data_list)
        except Exception as e:
            logger.error(f"Failed to collate data: {str(e)}")
            raise

def tree_to_graph(root, test_ratio=0):
    """Converts a search tree to a PyTorch Geometric graph."""
    if root is None:
        logger.warning("Received None root in tree_to_graph")
        return None

    try:
        nodes = []
        edges = []
        targets = []

        def traverse(node, parent_id=None):
            try:
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

                if parent_id is not None:
                    edges.append([parent_id, node_id])

                for child in node.children:
                    traverse(child, node_id)

            except AttributeError as e:
                logger.error(f"Node missing required attribute: {str(e)}")
                raise
            except Exception as e:
                logger.error(f"Error in traverse: {str(e)}")
                raise

        # Start traversal from the root
        traverse(root)

        if not nodes:
            logger.warning("No nodes were processed in tree_to_graph")
            return None

        # Convert data to torch tensors
        x = torch.tensor(nodes, dtype=torch.float)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
        y = torch.tensor(targets, dtype=torch.float)

        if test_ratio > 0:
            num_nodes = len(nodes)
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)

            train_indices = random.sample(
                range(num_nodes), int((1-test_ratio) * num_nodes))
            train_mask[train_indices] = True
            test_mask[~train_mask] = True

            return Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)
        else:
            return Data(x=x, edge_index=edge_index, y=y)

    except Exception as e:
        logger.error(f"Error in tree_to_graph: {str(e)}")
        return None

def save_processed_data(loader, save_path):
    """Save a processed DataLoader to disk."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Extract dataset from loader
        dataset = loader.dataset

        # Create save dictionary with all necessary information
        save_dict = {
            'slices': dataset.slices,
            'data': dataset.data,
            'batch_size': loader.batch_size,
            'shuffle': loader.shuffle,
        }

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
        dataset = InMemoryDataset(None, None, None)
        dataset.data = save_dict['data']
        dataset.slices = save_dict['slices']

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
        # If there's an error loading cached data, delete it
        try:
            load_path.unlink()
            logger.info(f"Deleted corrupted cache file: {load_path}")
        except:
            pass
        raise

def get_filtered_dataloaders(root_dir, processed_path=None, batch_size=32,
                           test_ratio=0.2, max_nodes=1000, max_depth=50,
                           max_branching=20, force_recache=False):
    """Get DataLoader with filtered data based on complexity criteria."""
    if processed_path:
        processed_path = Path(processed_path)

        # Check if we should use cached data
        if not force_recache and processed_path.exists():
            try:
                logger.info("Found processed data, loading from cache...")
                return load_processed_data(processed_path)
            except Exception as e:
                logger.warning(f"Failed to load cached data: {e}")
                logger.info("Will reprocess data...")

    logger.info("Creating new filtered DataLoader...")
    dataset = FilteredTreeDataset(
        root_dir=root_dir,
        max_nodes=max_nodes,
        max_depth=max_depth,
        max_branching=max_branching,
        test_ratio=test_ratio
    )
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

    # Use filtered dataloader with conservative limits
    loader = get_filtered_dataloaders(
        root_dir=data_dir,
        processed_path=processed_path,
        batch_size=32,
        test_ratio=0.2,
        max_nodes=1000,
        max_depth=50,
        max_branching=20
    )

    print(f"Dataset loaded with {len(loader)} batches")

if __name__ == '__main__':
    main()