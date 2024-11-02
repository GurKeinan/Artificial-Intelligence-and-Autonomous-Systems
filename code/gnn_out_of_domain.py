"""
This module trains and evaluates Graph Neural Networks (GNNs) on out-of-domain datasets.
The goal is to assess the generalization capability of GNNs when applied to datasets that differ from the training data.
"""

from pathlib import Path
import shutil
import sys
from tqdm import tqdm
import torch
from torch.optim.adamw import AdamW

from prepare_full_graph_dataset import FilteredTreeDataset, SerializableDataLoader,\
                                        load_processed_data, save_processed_data
from gnns import FullGraphsGNN
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
TEST_RATIO = 0.2
MAX_GRAD_NORM = 1.0

# Prefix constants
TRAIN_PREFIX = "sp"
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

def train_with_warmup(model, train_loader, eval_loader, optimizer, epochs, warmup_epochs, max_grad_norm):
    """
    Trains the model with a warmup phase.

    Args:
        model (torch.nn.Module): The GNN model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        eval_loader (torch.utils.data.DataLoader): DataLoader for the evaluation data.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        epochs (int): Total number of training epochs.
        warmup_epochs (int): Number of warmup epochs.
        max_grad_norm (float): Maximum gradient norm for clipping.

    Returns:
        None
    """


    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
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

        for batch in tqdm(train_loader, total=len(train_loader)):
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
            val_loss = evaluate(model, eval_loader)
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), repo_root / 'models' / f'gnn_ood_train_{TRAIN_PREFIX}_best_model.pth')

def evaluate(model, loader):
    """
    Evaluates the model on the given data loader.

    Args:
        model (torch.nn.Module): The GNN model to evaluate.
        loader (torch.utils.data.DataLoader): DataLoader for the evaluation data.

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
    train_loader, eval_loader = get_pruned_dataloaders(train_full_loader,
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
    test_loader = get_pruned_dataloaders(test_full_loader, test_ratio=0.0, logger=logger)[0]

    logger.info("Training dataset loaded with %d batches", len(train_loader))
    logger.info("Evaluation dataset loaded with %d batches", len(eval_loader))

    feature_dim = train_full_loader.dataset.num_features
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
        eval_loader,
        optimizer,
        epochs=EPOCHS,
        warmup_epochs=WARMUP_EPOCHS,
        max_grad_norm=MAX_GRAD_NORM
    )

    logger.info("Testing the best model on the other domain")
    ood_loss = evaluate(model, test_loader)
    logger.info("Error on the other domain: %f", ood_loss)

if __name__ == "__main__":
    main()
