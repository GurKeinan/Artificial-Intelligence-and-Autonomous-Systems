"""
This module trains and evaluates Graph Neural Networks (GNNs) on out-of-domain datasets. The goal is to assess the generalization capability of GNNs when applied to datasets that differ from the training data.
"""

import logging
from pathlib import Path
from datetime import datetime
import shutil
import sys
from tkinter.tix import MAX
from tqdm import tqdm
import torch
from torch.optim.adamw import AdamW
from prepare_full_graph_dataset import FilteredTreeDataset, SerializableDataLoader,\
                                        load_processed_data, save_processed_data
from gnns import FullGraphsGNN

# Constants
MAX_NODES = 15000
HIDDEN_DIM = 256
NUM_LAYERS = 4
HEADS = 4
DROPOUT = 0.2
LAYER_NORM = True
RESIDUAL_FREQUENCY = 2
MAX_GRAD_NORM = 1.0
LR = 0.001
WEIGHT_DECAY = 0.01
EPOCHS = 50
WARMUP_EPOCHS = 10
BATCH_SIZE = 16
TEST_RATIO = 0.2
TRAIN_PREFIX = "sp"
EVAL_PREFIX = "bw"

repo_root = Path(__file__).resolve().parent
dataset_creation_path = repo_root / "dataset_creation"
sys.path.append(str(dataset_creation_path))

# create models directory if it doesn't exist
models_dir = repo_root / "models"
models_dir.mkdir(exist_ok=True)

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Remove existing log file
log_filename = log_dir / f"gnn_ood__train_{TRAIN_PREFIX}_eval_{EVAL_PREFIX}.log"
if log_filename.exists():
    log_filename.unlink()

file_handler = logging.FileHandler(log_filename)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_with_warmup(model, loader, optimizer, epochs, warmup_epochs, max_grad_norm):
    """
    Trains the model with a warmup learning rate schedule.

    Args:
        model (torch.nn.Module): The model to train.
        loader (torch.utils.data.DataLoader): The data loader containing the training data.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        epochs (int): The number of epochs to train for.
        warmup_epochs (int, optional): The number of epochs to warm up the learning rate.
        max_grad_norm (float, optional): The maximum gradient norm for gradient clipping.
    """
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        epochs=epochs,
        steps_per_epoch=len(loader),
        pct_start=warmup_epochs/epochs
    )

    model = model.to(device)
    criterion = torch.nn.MSELoss()

    best_loss = float('inf')

    for epoch in range(epochs):
        logger.info("Starting Epoch %d", epoch + 1)
        model.train()
        total_loss = 0
        nodes_num = 0

        for batch in tqdm(loader, total=len(loader)):
            optimizer.zero_grad()
            batch = batch.to(device)
            predictions = model(batch)
            loss = criterion(predictions, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * len(batch.y)
            nodes_num += len(batch.y)


        avg_loss = total_loss / nodes_num
        logger.info('Epoch %d, Loss: %.4f, LR: %.6f',
                    epoch + 1, avg_loss, scheduler.get_last_lr()[0])

        if epoch % 5 == 0:
            val_loss = evaluate(model, loader)
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), repo_root / 'models' / f'gnn_ood_train_{TRAIN_PREFIX}_eval_{EVAL_PREFIX}_best_model.pth')

def evaluate(model, loader):
    """
    Evaluates the model on the provided data loader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        loader (torch.utils.data.DataLoader): The data loader containing the evaluation data.

    Returns:
        float: The average loss over the evaluation dataset.
    """
    model.eval()
    total_loss = 0
    num_nodes = 0
    criterion = torch.nn.MSELoss()

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predictions = model(batch)
            loss = criterion(predictions, batch.y)
            total_loss += loss.item() * len(batch.y)
            num_nodes += len(batch.y)

    avg_loss = total_loss / num_nodes
    logger.info('Evaluation average Loss: %.4f', avg_loss)
    return avg_loss

def filter_files_by_prefix(root_dir, prefix):
    """
    Filters files in the root directory by the given prefix.

    Args:
        root_dir (str): The root directory containing the dataset.
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
        root_dir (str): The root directory containing the dataset.
        processed_path (str): The path to save/load the processed data loader.
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

    bw_dir = base_dir  / "processed" / f"bw_dataloader_{MAX_NODES}.pt"
    train_loader = get_filtered_dataloaders_by_prefix(
        root_dir=data_dir,
        processed_path=bw_dir,
        prefix=TRAIN_PREFIX,
        batch_size=BATCH_SIZE,
        test_ratio=0.0,
        max_nodes=MAX_NODES
    )

    sp_dir = base_dir / "processed" / f"sp_dataloader_{MAX_NODES}.pt"
    eval_loader = get_filtered_dataloaders_by_prefix(
        root_dir=data_dir,
        processed_path=sp_dir,
        prefix=EVAL_PREFIX,
        batch_size=BATCH_SIZE,
        test_ratio=0.0,
        max_nodes=MAX_NODES
    )

    logger.info("Training dataset loaded with %d batches", len(train_loader))
    logger.info("Evaluation dataset loaded with %d batches", len(eval_loader))

    feature_dim = train_loader.dataset.num_features
    model = FullGraphsGNN(
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
        optimizer,
        epochs=EPOCHS,
        warmup_epochs=WARMUP_EPOCHS,
        max_grad_norm=MAX_GRAD_NORM
    )

    logger.info("Evaluating the model on the evaluation dataset...")
    evaluate(model, eval_loader)

if __name__ == "__main__":
    main()
