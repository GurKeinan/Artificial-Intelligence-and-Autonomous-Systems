import random
import torch
from torch_geometric.data import Data
import logging
from prepare_graph_dataset import SerializableDataLoader


def prune_graph_nodes(data):
    """
    Prune nodes from a graph based on random threshold selection.

    Args:
        data (Data): PyTorch Geometric Data object containing the graph

    Returns:
        Data: Pruned graph with subset of nodes
    """
    # Choose random threshold from [0.3, 0.5, 0.7]
    threshold = random.choice([0.3, 0.5, 0.7])

    total_nodes = data.num_nodes
    # Create mask for nodes to keep based on their indices
    keep_mask = torch.arange(total_nodes) / total_nodes < threshold

    # Get node indices to keep
    node_idx = torch.where(keep_mask)[0]

    # Update edge indices to only include edges between remaining nodes
    edge_mask = torch.isin(data.edge_index[0], node_idx) & torch.isin(data.edge_index[1], node_idx)
    new_edge_index = data.edge_index[:, edge_mask]

    # Create node index mapping for updating edge indices
    idx_map = {int(old_idx): new_idx for new_idx, old_idx in enumerate(node_idx)}
    new_edge_index = torch.tensor([[idx_map[int(i)] for i in new_edge_index[0]],
                                  [idx_map[int(i)] for i in new_edge_index[1]]])

    # Create new pruned graph
    pruned_data = Data(
        x=data.x[keep_mask],
        edge_index=new_edge_index,
        y=data.y[keep_mask],
    )

    return pruned_data

def get_pruned_dataloaders(loader, train_ratio, eval_ratio, test_ratio, logger):
    """
    Prune nodes from graphs in a PyTorch Geometric DataLoader.
    Samples a fraction from [0.3, 0.5, 0.7], nodes with (serial_number / total_nodes) < threshold are kept.

    Args:
        loader (DataLoader): PyTorch Geometric DataLoader
        train_ratio (float): Fraction of nodes to assign to training set
        eval_ratio (float): Fraction of nodes to assign to evaluation set
        test_ratio (float): Fraction of nodes to assign to test set
        logger: Logger object for logging messages

    Returns:
        train_loader (DataLoader): DataLoader with pruned graphs for training
        eval_loader (DataLoader): DataLoader with pruned graphs for evaluation
        test_loader (DataLoader or None): DataLoader with pruned graphs for testing, or None if test_ratio is 0
    """

    # Apply pruning to dataset
    pruned_dataset = []
    total_original_nodes = 0
    total_pruned_nodes = 0

    for data in loader.dataset:
        total_original_nodes += data.num_nodes
        pruned_data = prune_graph_nodes(data)
        total_pruned_nodes += pruned_data.num_nodes
        pruned_dataset.append(pruned_data)

    # Create new dataset with pruned graphs
    dataset = pruned_dataset
    logger.info("Total nodes in original dataset: %d", total_original_nodes)
    logger.info("Total nodes in pruned dataset: %d", total_pruned_nodes)

    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    eval_size = int(eval_ratio * total_size)
    test_size = total_size - train_size - eval_size

    if test_ratio == 0:
        if eval_ratio == 0:
            train_dataset = dataset
            train_loader = SerializableDataLoader(
                train_dataset,
                batch_size=loader.batch_size,
                shuffle=True
            )
            eval_loader = None
            test_loader = None

        else:
            train_dataset, eval_dataset = torch.utils.data.random_split(
                dataset, [train_size, eval_size], generator=torch.Generator().manual_seed(42)
            )
            train_loader = SerializableDataLoader(
                train_dataset,
                batch_size=loader.batch_size,
                shuffle=True
            )
            eval_loader = SerializableDataLoader(
                eval_dataset,
                batch_size=loader.batch_size,
                shuffle=False
            )
            test_loader = None
    else:
        train_dataset, eval_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, eval_size, test_size], generator=torch.Generator().manual_seed(42)
        )
        train_loader = SerializableDataLoader(
            train_dataset,
            batch_size=loader.batch_size,
            shuffle=True
        )
        eval_loader = SerializableDataLoader(
            eval_dataset,
            batch_size=loader.batch_size,
            shuffle=False
        )
        test_loader = SerializableDataLoader(
            test_dataset,
            batch_size=loader.batch_size,
            shuffle=False
        )


    feature_dim = loader.dataset.num_features
    # logs
    logger.info("Training dataset loaded with %d batches", len(train_loader))
    if eval_loader is not None:
        logger.info("Evaluation dataset loaded with %d batches", len(eval_loader))
    else:
        logger.info("Evaluation dataset not created as eval_ratio is 0")
    if test_loader is not None:
        logger.info("Test dataset loaded with %d batches", len(test_loader))
    else:
        logger.info("Test dataset not created as test_ratio is 0")
    logger.info("Feature dimension: %d", feature_dim)
    logger.info("Total nodes in training dataset: %d", sum([data.num_nodes for data in train_loader.dataset]))
    if eval_loader is not None:
        logger.info("Total nodes in evaluation dataset: %d", sum([data.num_nodes for data in eval_loader.dataset]))
    else:
        logger.info("Total nodes in evaluation dataset: 0")
    if test_loader is not None:
        logger.info("Total nodes in test dataset: %d", sum([data.num_nodes for data in test_loader.dataset]))

    return train_loader, eval_loader, test_loader




def setup_logger(logfile_path):
    """
    Setup logger to log messages to console and file.

    Args:
        logfile_path (Path): Path to log file

    Returns:
        logger: Logger object
    """
    # Create logs directory if it doesn't exist
    log_dir = logfile_path.parent
    log_dir.mkdir(exist_ok=True)

    # Remove existing log file
    if logfile_path.exists():
        logfile_path.unlink()

    file_handler = logging.FileHandler(logfile_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Set up logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    return logger
