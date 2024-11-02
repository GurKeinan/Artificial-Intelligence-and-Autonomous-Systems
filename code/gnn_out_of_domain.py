"""
This module trains and evaluates Graph Neural Networks (GNNs) on out-of-domain datasets.
The goal is to assess the generalization capability of GNNs when applied to datasets that differ from the training data.
"""

from pathlib import Path
import shutil
import sys
import torch
from torch.optim.adamw import AdamW

from gnn_training_evaluating import evaluate, train_with_warmup
from prepare_graph_dataset import FilteredTreeDataset, SerializableDataLoader,\
                                        load_processed_data, save_processed_data
from gnn_architectures import HeavyGNN
from utils import setup_logger, get_pruned_dataloaders

# Filtering constants
MAX_NODES = 15000
# Model constants
HIDDEN_DIM = 256
NUM_LAYERS = 4
HEADS = 4
DROPOUT = 0.2
LAYER_NORM = True
RESIDUAL_FREQUENCY = 2
# Optimizer constants
LR = 0.001
WEIGHT_DECAY = 0.01
# Training constants
EPOCHS = 50
WARMUP_EPOCHS = 10
BATCH_SIZE = 16
TRAIN_RATIO = 0.8
EVAL_RATIO = 0.2
TEST_RATIO = 0.0
MAX_GRAD_NORM = 1.0
PATIENCE = 10
EVAL_EVERY = 1

# Prefix constants
TRAIN_PREFIX = "bw"
EVAL_PREFIX = "sp" if TRAIN_PREFIX == "bw" else "bw"

repo_root = Path(__file__).resolve().parent
dataset_creation_path = repo_root / "dataset_creation"
sys.path.append(str(dataset_creation_path))

# create models directory if it doesn't exist
models_dir = repo_root / "models"
models_dir.mkdir(exist_ok=True)

# Set up logging
logfile_path = repo_root / "logs" / f"gnn_out_of_domain_train_{TRAIN_PREFIX}_eval_{EVAL_PREFIX}.log"
logger = setup_logger(logfile_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def filter_files_by_prefix(root_dir, prefix):
    """
    Filters files in the root directory by the given prefix.

    Args:
        root_dir (Path): The root directory containing the dataset.
        prefix (str): The prefix to filter files.

    Returns:
        list: A list of filtered file paths that match the given prefix.
    """
    root_path = Path(root_dir)
    filtered_files = []
    for dir_path in root_path.iterdir():
        if dir_path.is_dir() and dir_path.name.startswith(prefix):
            filtered_files.extend([str(p) for p in dir_path.rglob('*.pkl')])
    return filtered_files

def get_filtered_dataloaders_by_prefix(root_dir, processed_path, prefix,batch_size, test_ratio, max_nodes):
    """
    Retrieves filtered data loaders by prefix.

    Args:
        root_dir (Path): The root directory containing the dataset.
        processed_path (Path): The path to save/load the processed data loader.
        prefix (str): The prefix to filter files.
        batch_size (int): The batch size for the data loader.
        test_ratio (float): The ratio of the dataset to use for testing.
        max_nodes (int): The maximum number of nodes in the dataset.

    Returns:
        SerializableDataLoader: The data loader containing the filtered dataset.
    """
    if processed_path and Path(processed_path).exists():
        logger.info("Loading DataLoader from %s", processed_path)
        loader = load_processed_data(processed_path)
        return loader
    else:
        logger.info("Filtering files with prefix %s", prefix)
        filtered_files = filter_files_by_prefix(root_dir, prefix)
        # Save filtered files to a directory
        filtered_files_path = Path(f"filtered_files_{prefix}")
        filtered_files_path.mkdir(exist_ok=True)
        for file in filtered_files:
            destination = filtered_files_path / Path(file).name
            shutil.copy(file, destination)

        dataset = FilteredTreeDataset(root_dir=filtered_files_path,
                                    max_nodes=max_nodes, test_ratio=test_ratio)
        loader = SerializableDataLoader(dataset, batch_size=batch_size, shuffle=True)

        logger.info("Saving DataLoader to %s", processed_path)
        save_processed_data(loader, processed_path)

        # delete the filtered files
        shutil.rmtree(filtered_files_path)
        return loader

def main():
    """
    Main function to set up and train a Graph Neural Network (GNN) model on out-of-domain datasets.
    """

    torch.manual_seed(42)
    logger.info("Using device: %s", device)

    base_dir = Path(__file__).resolve().parent
    if base_dir.name != "code":
        base_dir = base_dir / "code"
    data_dir = base_dir / "dataset"

    if TRAIN_PREFIX == "bw":
        train_domain_dir = base_dir  / "processed" / f"bw_dataloader_{MAX_NODES}.pt"
        eval_domain_dir = base_dir  / "processed" / f"sp_dataloader_{MAX_NODES}.pt"
    else:
        train_domain_dir = base_dir  / "processed" / f"sp_dataloader_{MAX_NODES}.pt"
        eval_domain_dir = base_dir  / "processed" / f"bw_dataloader_{MAX_NODES}.pt"

    train_full_loader = get_filtered_dataloaders_by_prefix(
        root_dir=data_dir,
        processed_path=train_domain_dir,
        prefix=TRAIN_PREFIX,
        batch_size=BATCH_SIZE,
        test_ratio=TEST_RATIO,
        max_nodes=MAX_NODES
    )
    train_loader, eval_loader, test_loader = get_pruned_dataloaders(train_full_loader,
                                                                    train_ratio=TRAIN_RATIO,
                                                                    eval_ratio=EVAL_RATIO,
                                                                    test_ratio=TEST_RATIO,
                                                                    logger=logger)

    test_full_loader = get_filtered_dataloaders_by_prefix(
        root_dir=data_dir,
        processed_path=eval_domain_dir,
        prefix=EVAL_PREFIX,
        batch_size=BATCH_SIZE,
        test_ratio=0.0,
        max_nodes=MAX_NODES
    )
    test_loader = get_pruned_dataloaders(test_full_loader, train_ratio=1.0, eval_ratio=0.0, test_ratio=0.0, logger=logger)[0]

    feature_dim = train_full_loader.dataset.num_features
    model = HeavyGNN(
        input_dim=feature_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        heads=HEADS,
        dropout=DROPOUT,
        layer_norm=LAYER_NORM,
        residual_frequency=RESIDUAL_FREQUENCY
    )

    optimizer = AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    logger.info("Starting training...")
    train_with_warmup(
        model,
        train_loader,
        eval_loader,
        optimizer,
        epochs=EPOCHS,
        warmup_epochs=WARMUP_EPOCHS,
        max_grad_norm=MAX_GRAD_NORM,
        patience=PATIENCE,
        eval_every=EVAL_EVERY,
        best_model_path=models_dir / f"gnn_ood_{TRAIN_PREFIX}_best_model.pth",
        device=device,
        logger=logger
    )

    logger.info("Testing the best model on the other domain")
    ood_loss = evaluate(model, test_loader, device)
    logger.info("Error on the other domain: %f", ood_loss)

if __name__ == "__main__":
    main()
