"""
This module provides functionality for loading, filtering, and processing search trees
into PyTorch Geometric graphs. It includes classes and functions for dataset handling,
data loading, and serialization.

Classes:
    SerializableDataLoader: A wrapper class to make DataLoader serializable.
    FilteredTreeDataset: An InMemoryDataset that filters trees based on node count.

Functions:
    tree_to_graph(root, test_ratio=0): Converts a search tree to a PyTorch Geometric graph.
    save_processed_data(loader, save_path): Saves a processed DataLoader to disk.
    load_processed_data(load_path): Loads a processed DataLoader from disk.
    get_filtered_dataloaders(root_dir, processed_path=None, batch_size=32,
    test_ratio=0.2, max_nodes=1000, force_recache=False):
        Returns a DataLoader with filtered data based on complexity criteria.
    main(): Main function to load the dataset and create a DataLoader.
"""
import pickle
from pathlib import Path
import random
import logging
from tqdm import tqdm
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader

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

class FilteredTreeDataset(InMemoryDataset): # pylint: disable=abstract-method
    """
    FilteredTreeDataset is a custom dataset class for loading and filtering tree-structured data.
    Attributes:
        root_dir (str): The root directory where the dataset is stored.
        max_nodes (int): The maximum number of nodes allowed in a tree
        for it to be included in the dataset.
        test_ratio (float): The ratio of the dataset to be used for testing.
        transform (callable, optional): A function/transform that takes in a data object
        and returns a transformed version.
        pre_transform (callable, optional): A function/transform that takes in a data object
        and returns a transformed version before any other processing.
    Methods:
        analyze_tree(root):
            Analyze a search tree for its properties.
            Args:
                root (TreeNode): The root node of the tree.
            Returns:
                int: The number of nodes in the tree.
        is_tree_acceptable(root):
            Check if a tree meets the criteria for inclusion.
            Args:
                root (TreeNode): The root node of the tree.
            Returns:
                bool: True if the tree meets the criteria, False otherwise.
        load_data():
            Load and process the dataset from the root directory.
            Returns:
                tuple: A tuple containing the data and slices.
            Raises:
                ValueError: If no valid data was loaded from the dataset.
    """

    def __init__(self, root_dir, max_nodes, test_ratio=0.2, transform=None, pre_transform=None):
        self.max_nodes = max_nodes
        super(FilteredTreeDataset, self).__init__(root_dir, transform, pre_transform)
        self.root_dir = root_dir
        self.test_ratio = test_ratio
        try:
            self.data, self.slices = self.load_data()
        except (ValueError, RuntimeError) as e:
            logger.error("Failed to load dataset: %s", str(e))
            raise

    def analyze_tree(self, root):
        """Analyze a search tree for its properties."""
        num_nodes = 0

        def traverse(node, depth=0):
            nonlocal num_nodes
            num_nodes += 1
            for child in node.children:
                traverse(child, depth + 1)

        traverse(root)
        return num_nodes

    def is_tree_acceptable(self, root):
        """Check if a tree meets our criteria for inclusion."""
        num_nodes = self.analyze_tree(root)

        # Log the tree properties for monitoring
        logger.debug("Tree properties - Nodes: %d", num_nodes)

        return num_nodes <= self.max_nodes

    def _log_processing_summary(self, failed_files):
        """Log summary of data processing results"""
        if failed_files:
            logger.warning("Failed to process the following files:")
            for file, reason in failed_files:
                logger.warning("  - %s: %s", file, reason)


    def _log_datasets(self, root_path):
        datasets = '\n'.join([str(p.as_posix()) for p in root_path.iterdir()
                              if p.name != '.DS_Store'])
        logger.info("Read Datasets: %s", datasets)

    def _count_total_files(self, root_path):
        total_files = sum(1 for _ in root_path.rglob('*.pkl'))
        logger.info("Found %d PKL files", total_files)
        return total_files

    def _process_file(self, pkl_file, data_list, failed_files):
        try:
            if pkl_file.stat().st_size == 0:
                self._handle_empty_file(pkl_file, failed_files)
                return

            tree = self._load_pickle_file(pkl_file, failed_files)
            if tree is None:
                return

            if not self.is_tree_acceptable(tree):
                logger.debug("Rejected tree from %s - too complex", pkl_file)
                return

            self._convert_tree_to_graph(pkl_file, tree, data_list, failed_files)

        except (EOFError, pickle.UnpicklingError, RuntimeError) as e:
            logger.warning("Failed to process file %s : %s", pkl_file, str(e))
            failed_files.append((pkl_file, f"Process error: {str(e)}"))

    def _handle_empty_file(self, pkl_file, failed_files):
        logger.warning("Skipping empty file: %s", pkl_file)
        failed_files.append((pkl_file, "Empty file"))

    def _load_pickle_file(self, pkl_file, failed_files):
        try:
            with pkl_file.open('rb') as f:
                return pickle.load(f)
        except (EOFError, pickle.UnpicklingError) as e:
            logger.warning("Failed to load corrupted pickle file %s: %s", pkl_file, str(e))
            failed_files.append((pkl_file, f"Pickle error: {str(e)}"))
            return None

    def _convert_tree_to_graph(self, pkl_file, tree, data_list, failed_files):
        try:
            graph_data = tree_to_graph(tree, self.test_ratio)
            if graph_data is not None:
                data_list.append(graph_data)
            else:
                logger.warning("Skipping %s: tree_to_graph returned None", pkl_file)
                failed_files.append((pkl_file, "Graph conversion failed"))
        except (ValueError, TypeError, AttributeError) as e:
            logger.warning("Failed to convert tree to graph for %s: %s", pkl_file, str(e))
            failed_files.append((pkl_file, f"Conversion error: {str(e)}"))

    def _collate_data(self, data_list):
        try:
            return self.collate(data_list)
        except Exception as e:
            logger.error("Failed to collate data: %s", e)
            raise

    def load_data(self):
        """
        Loads data from pickle files in the specified root directory.
        This method processes all `.pkl` files found in the root directory
        and its subdirectories.
        It logs the datasets, counts the total number of files,
        and processes each file to load the data.
        It also keeps track of the number of accepted and rejected files,
        and logs a summary of the processing.
        Raises:
            ValueError: If no valid data is loaded from the dataset.
        Returns:
            list: A collated list of data loaded from the pickle files.
        """

        data_list = []
        failed_files = []
        root_path = Path(self.root_dir)

        self._log_datasets(root_path)
        total_files = self._count_total_files(root_path)

        for pkl_file in tqdm(root_path.rglob('*.pkl'), total=total_files):
            self._process_file(pkl_file, data_list, failed_files)


        if not data_list:
            raise ValueError("No valid data was loaded from the dataset")

        return self._collate_data(data_list)



def tree_to_graph(root, test_ratio=0.2):
    """Converts a search tree to a PyTorch Geometric graph."""
    if root is None:
        logger.warning("Received None root in tree_to_graph")
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
        targets.append(node.progress)

        if parent_id is not None:
            edges.append([parent_id, node_id])

        for child in node.children:
            traverse(child, node_id)



    # Start traversal from the root
    traverse(root)

    if not nodes:
        logger.warning("No nodes were processed in tree_to_graph")
        return None

    # Convert data to torch tensors
    x = torch.tensor(nodes, dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges \
    else torch.empty((2, 0), dtype=torch.long)
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
        logger.info("Successfully saved processed data to %s", save_path)
    except Exception as e:
        logger.error("Error saving data: %s", e)
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

        logger.info("Successfully loaded processed data from %s", load_path)
        return loader

    except Exception as e:
        logger.error("Error loading data: %s", e)
        # If there's an error loading cached data, delete it
        try:
            load_path.unlink()
            logger.info("Deleted corrupted cache file: %s", load_path)
        except (OSError, IOError, pickle.UnpicklingError):
            pass
        raise

def get_filtered_dataloaders(root_dir, processed_path, batch_size=32,
                           test_ratio=0.2, max_nodes=1000, force_recache=False):
    """Get DataLoader with filtered data based on complexity criteria.
    Parameters:
        root_dir (Path): Path to the root directory of the dataset.
        processed_path (Path): Path to save/load processed data.
        batch_size (int): Batch size for DataLoader.
        test_ratio (float): Ratio of data to use for testing.
        max_nodes (int): Maximum number of nodes in a tree.
        force_recache (bool): Whether to force reprocessing of data.
    Returns:
        DataLoader: DataLoader with filtered data based on complexity criteria.
    """
    if processed_path:
        processed_path = Path(processed_path)

        # Check if we should use cached data
        if not force_recache and processed_path.exists():
            try:
                logger.info("Found processed data, loading from cache...")
                return load_processed_data(processed_path)
            except (pickle.UnpicklingError, OSError) as e:
                logger.warning("Failed to load cached data: %s", e)
                logger.info("Will reprocess data...")

    logger.info("Creating new filtered DataLoader...")
    dataset = FilteredTreeDataset(
        root_dir=root_dir,
        max_nodes=max_nodes,
        test_ratio=test_ratio
    )
    loader = SerializableDataLoader(dataset, batch_size=batch_size, shuffle=True)

    if processed_path:
        # Save for future use
        save_processed_data(loader, processed_path)

    return loader

def main():
    """
    Main function to load and process the dataset.
    This function sets up the base directory, data directory, and processed data path.
    It then uses a filtered dataloader with specified parameters to load the dataset.
    The function performs the following steps:
    1. Resolves the base directory of the script.
    2. Ensures the base directory is set to "code" if not already.
    3. Defines the data directory and processed data path.
    4. Loads the dataset using a filtered dataloader with conservative limits.
    5. Prints the number of batches loaded.
    Parameters:
    None
    Returns:
    None
    """

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
        max_nodes=1000
    )

    print(f"Dataset loaded with {len(loader)} batches")

if __name__ == '__main__':
    main()
