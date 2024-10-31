"""
This script trains a Graph Neural Network (GNN) using dynamically sampled subgraphs
from a larger graph dataset.
It includes the following components:

- Constants for filtering, sampling, model configuration, and training.
- Logging setup to record training progress and results.
- `DynamicSampledLoader` class to dynamically sample subgraphs during training.
- `SampleGNN` class defining the GNN architecture.
- `train_sampled_gnn` function to train the GNN using the dynamically sampled subgraphs.
- `evaluate_sampled_model` function to evaluate the GNN on sampled subgraphs.
- `main` function to load data, initialize the model, and start the training process.

Classes:
    DynamicSampledLoader: DataLoader that dynamically samples subgraphs during iteration.
    SampleGNN: Modified GNN to handle sampled subgraphs.

Functions:
    train_sampled_gnn(model, original_loader, optimizer, epochs,
    warmup_epochs=10, max_grad_norm=1.0):
        Train the GNN using dynamically sampled subgraphs.
    evaluate_sampled_model(model, loader, mask_type="Test"):
        Evaluate the model on sampled subgraphs.
    main():
        Main function to load data, initialize the model, and start the training process.

Constants:
    MAX_NODES: Maximum number of nodes in a graph.
    SAMPLES_PER_EPOCH: Total samples to generate per epoch.
    MAX_DISTANCE: Maximum hop distance for subgraphs.
    NUM_GNN_LAYERS: Number of GNN layers based on max distance.
    HIDDEN_DIM: Hidden dimension size for the GNN.
    DROPOUT: Dropout rate.
    LAYER_NORM: Boolean indicating whether to use layer normalization.
    RESIDUAL_FREQUENCY: Frequency of residual connections.
    LR: Learning rate.
    WEIGHT_DECAY: Weight decay for the optimizer.
    EPOCHS: Number of training epochs.
    WARMUP_EPOCHS: Number of warmup epochs for the learning rate scheduler.
    BATCH_SIZE: Batch size for training.
    TEST_RATIO: Ratio of the dataset to use for testing.
"""
import logging
import random
from pathlib import Path
from datetime import datetime
import sys
import torch
from torch.optim.adamw import AdamW
from torch_geometric.data import Data, Batch
from torch_geometric.utils import k_hop_subgraph
from tqdm import tqdm
from prepare_full_graph_dataset import get_filtered_dataloaders
from gnns import SampleGNN

# constants for filtering
MAX_NODES = 10000  # Maximum number of nodes in a graph

# Constants for sampling
SAMPLES_PER_EPOCH = 10000  # Total samples to generate per epoch
MAX_DISTANCE = 3  # Maximum hop distance for subgraphs
NUM_GNN_LAYERS = MAX_DISTANCE + 2  # Number of GNN layers based on max distance

# Model constants
HIDDEN_DIM = 256

DROPOUT = 0.2
LAYER_NORM = True
RESIDUAL_FREQUENCY = 2

# Training constants
LR = 0.001
WEIGHT_DECAY = 0.01
EPOCHS = 100
WARMUP_EPOCHS = 10
BATCH_SIZE = 128
TEST_RATIO = 0.2

# attach dataset_creation to sys.path for habndling pickles if necessary
repo_root = Path(__file__).resolve().parent
dataset_creation_path = repo_root / "dataset_creation"
sys.path.append(str(dataset_creation_path))

# Set up logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime('%d.%m.%Y_%H:%M:%S')
log_filename = log_dir / f"training_sampled_GNN_{timestamp}.log"

file_handler = logging.FileHandler(log_filename)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DynamicSampledLoader:
    """
    A data loader that dynamically samples subgraphs from a given dataset of graphs.
    Attributes:
        original_dataset (Dataset): The original dataset containing the graphs.
        samples_per_epoch (int): The number of samples to generate per epoch.
        max_distance (int): The maximum distance for k-hop subgraph sampling.
        batch_size (int): The number of samples per batch.
        shuffle (bool): Whether to shuffle the samples in each batch.
    Methods:
        sample_subgraph(graph):
            Samples a single subgraph from a given graph.
        __iter__():
            Iterates over the dataset, yielding batches of sampled subgraphs.
        __len__():
            Returns the number of batches per epoch.
    """

    def __init__(self, original_dataset, samples_per_epoch, max_distance,
                 batch_size, shuffle=True):
        self.original_dataset = original_dataset
        self.samples_per_epoch = samples_per_epoch
        self.max_distance = max_distance
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Cache dataset indices for efficient sampling
        self.dataset_indices = list(range(len(original_dataset)))

    def sample_subgraph(self, graph):
        """
        Samples a k-hop subgraph from the given graph.
        Parameters:
        -----------
        graph : torch_geometric.data.Data
            The input graph from which to sample the subgraph.
            It is expected to have attributes `x` (node features),
            `edge_index` (edge list), and optionally `y` (node labels), `train_mask`, `test_mask`.
        Returns:
        --------
        torch_geometric.data.Data
            A new graph object containing the sampled subgraph with the following attributes:
            - `x`: Node features of the subgraph.
            - `edge_index`: Edge list of the subgraph.
            - `y`: Node labels of the subgraph (if present in the original graph).
            - `train_mask`: Training mask for the subgraph (if present in the original graph).
            - `test_mask`: Test mask for the subgraph (if present in the original graph).
            - `center_node`: A tensor containing the index of the central node
            (always 0 after relabeling).
        """

        num_nodes = graph.x.size(0)
        node_idx = random.randrange(num_nodes)

        # Get k-hop subgraph
        subset, edge_index, _, _ = k_hop_subgraph(
            node_idx=node_idx,
            num_hops=self.max_distance,
            edge_index=graph.edge_index,
            relabel_nodes=True,
            num_nodes=num_nodes
        )

        # Create new graph with the subgraph data
        x = graph.x[subset]
        y = graph.y[subset]

        # Create train/test masks if they exist in original graph
        train_mask = graph.train_mask[subset] if hasattr(graph, 'train_mask') else None
        test_mask = graph.test_mask[subset] if hasattr(graph, 'test_mask') else None

        return Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            test_mask=test_mask,
            center_node=torch.tensor([0])  # The central node is always 0 after relabeling
        )

    def __iter__(self):
        sampled_graphs = []
        samples_remaining = self.samples_per_epoch

        # Sample graphs and nodes until we have enough samples
        while samples_remaining > 0:
            # Sample a graph (with replacement)
            graph_idx = random.choice(self.dataset_indices)
            graph = self.original_dataset[graph_idx]

            # Sample a subgraph
            sampled_graphs.append(self.sample_subgraph(graph))
            samples_remaining -= 1

            # Create and yield a batch when we have enough samples
            if len(sampled_graphs) >= self.batch_size:
                if self.shuffle:
                    random.shuffle(sampled_graphs[:self.batch_size])
                yield Batch.from_data_list(sampled_graphs[:self.batch_size])
                sampled_graphs = sampled_graphs[self.batch_size:]

        # Handle remaining samples
        if sampled_graphs:
            if self.shuffle:
                random.shuffle(sampled_graphs)
            yield Batch.from_data_list(sampled_graphs)

    def __len__(self):
        return (self.samples_per_epoch + self.batch_size - 1) // self.batch_size

def initialize_scheduler(optimizer, epochs, warmup_epochs, loader):
    """
    Initializes and returns a OneCycleLR scheduler for the given optimizer.
    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to schedule the learning rate.
        epochs (int): The total number of epochs for training.
        warmup_epochs (int): The number of epochs to warm up the learning rate.
        loader (torch.utils.data.DataLoader): DataLoader providing the training data.
    Returns:
        torch.optim.lr_scheduler.OneCycleLR: The initialized learning rate scheduler.
    """

    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        epochs=epochs,
        steps_per_epoch=len(loader),
        pct_start=warmup_epochs / epochs
    )

def train_one_epoch(model, loader, optimizer, scheduler, criterion, max_grad_norm):
    """
    Trains the model for one epoch.
    Args:
        model (torch.nn.Module): The model to be trained.
        loader (torch.utils.data.DataLoader): DataLoader providing the training data.
        optimizer (torch.optim.Optimizer): Optimizer for updating the model parameters.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        criterion (torch.nn.Module): Loss function.
        max_grad_norm (float): Maximum gradient norm for gradient clipping.
    Returns:
        float: The average loss over all valid batches.
    """

    model.train()
    total_loss = 0
    valid_batches = 0

    for batch in tqdm(loader, total=len(loader)):
        optimizer.zero_grad()
        batch = batch.to(device)
        predictions = model(batch)

        if hasattr(batch, 'train_mask'):
            train_pred = predictions[batch.train_mask]
            train_true = batch.y[batch.train_mask]

            if len(train_pred) > 0:
                loss = criterion(train_pred, train_true)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                valid_batches += 1

    avg_loss = total_loss / valid_batches if valid_batches > 0 else float('inf')
    return avg_loss

def evaluate_sampled_model(model, loader, mask_type="Test"):
    """
    Evaluates a sampled model on a given dataset loader.
    Args:
        model (torch.nn.Module): The model to be evaluated.
        loader (torch.utils.data.DataLoader): DataLoader containing the dataset to evaluate.
        mask_type (str, optional): Type of mask to use for evaluation.
                                   Options are "Train", "Test", or "Full". Default is "Test".
    Returns:
        float or None: The average loss over the samples if there are any, otherwise None.
    """

    model.eval()
    total_loss = 0
    num_samples = 0
    criterion = torch.nn.MSELoss()

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predictions = model(batch)

            if mask_type == "Train":
                mask = batch.train_mask
            elif mask_type == "Test":
                mask = batch.test_mask
            else:  # "Full"
                mask = torch.ones_like(batch.train_mask)

            if mask.sum() > 0:
                loss = criterion(predictions[mask], batch.y[mask])
                total_loss += loss.item() * mask.sum()
                num_samples += mask.sum()

    if num_samples > 0:
        avg_loss = total_loss / num_samples
        logger.info('%s Average Loss: %.4f', mask_type, avg_loss)
        return avg_loss
    return None

def evaluate_and_save_model(model, loader, best_val_loss):
    """
    Evaluates the given model using the provided data loader and saves the model
    if it achieves a new best validation loss.
    Args:
        model (torch.nn.Module): The model to be evaluated and potentially saved.
        loader (torch.utils.data.DataLoader): The data loader providing the dataset for evaluation.
        best_val_loss (float): The current best validation loss to compare against.
    Returns:
        float: The updated best validation loss.
    """

    evaluate_sampled_model(model, loader, "Train")
    val_loss = evaluate_sampled_model(model, loader, "Test")

    if val_loss is not None and val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_sample_training_model.pth')
        logger.info('New best model saved with validation loss: %.4f', best_val_loss)

    return best_val_loss

def train_sampled_gnn(model, original_loader, optimizer, epochs,
                      warmup_epochs=10, max_grad_norm=1.0):
    """
    Trains a Graph Neural Network (GNN) model using dynamic sampling.
    Args:
        model (torch.nn.Module): The GNN model to be trained.
        original_loader (torch.utils.data.DataLoader): DataLoader for the original dataset.
        optimizer (torch.optim.Optimizer): Optimizer for training the model.
        epochs (int): Number of epochs to train the model.
        warmup_epochs (int, optional): Number of warmup epochs for the learning rate scheduler.
        Default is 10.
        max_grad_norm (float, optional): Maximum norm for gradient clipping. Default is 1.0.
    Returns:
        None
    """

    model = model.to(device)
    criterion = torch.nn.MSELoss()

    # Create sampled loader
    sampled_loader = DynamicSampledLoader(
        original_dataset=original_loader.dataset,
        samples_per_epoch=SAMPLES_PER_EPOCH,
        max_distance=MAX_DISTANCE,
        batch_size=BATCH_SIZE
    )

    scheduler = initialize_scheduler(optimizer, epochs, warmup_epochs, sampled_loader)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        logger.info("Starting Epoch %d", epoch + 1)
        avg_loss = train_one_epoch(model, sampled_loader,
                                   optimizer, scheduler, criterion, max_grad_norm)
        logger.info('Epoch %d, Loss: %.4f, LR: %.6f',
                    epoch + 1, avg_loss, scheduler.get_last_lr()[0])

        if epoch % 5 == 0:
            best_val_loss = evaluate_and_save_model(model, sampled_loader, best_val_loss)



def main():
    """
    Main function to train and evaluate a Sampled Graph Neural Network (GNN).
    This function performs the following steps:
    1. Sets the random seed for reproducibility.
    2. Logs the device being used for computation.
    3. Loads the dataset with a specified batch size.
    4. Initializes the Sampled GNN model with specified hyperparameters.
    5. Sets up the optimizer for training.
    6. Trains the model using dynamic sampling.
    7. Loads the best model from the training process.
    8. Evaluates the best model on the entire dataset.
    The function relies on several global constants and functions:
    - `MAX_NODES`: Maximum number of nodes in the graph.
    - `BATCH_SIZE`: Batch size for data loading.
    - `TEST_RATIO`: Ratio of the dataset to be used for testing.
    - `HIDDEN_DIM`: Dimension of the hidden layers in the GNN.
    - `NUM_GNN_LAYERS`: Number of layers in the GNN.
    - `DROPOUT`: Dropout rate for the GNN layers.
    - `LAYER_NORM`: Whether to use layer normalization.
    - `RESIDUAL_FREQUENCY`: Frequency of residual connections in the GNN.
    - `LR`: Learning rate for the optimizer.
    - `WEIGHT_DECAY`: Weight decay for the optimizer.
    - `EPOCHS`: Number of training epochs.
    - `WARMUP_EPOCHS`: Number of warmup epochs for training.
    - `get_filtered_dataloaders`: Function to load and filter the dataset.
    - `SampleGNN`: Class representing the Sampled GNN model.
    - `train_sampled_gnn`: Function to train the Sampled GNN model.
    - `evaluate_sampled_model`: Function to evaluate the trained model.
    Returns:
        None
    """

    torch.manual_seed(42)
    logger.info("Using device: %s", device)

    # Load data with smaller batch size
    base_dir = Path(__file__).resolve().parent
    if base_dir.name != "code":
        base_dir = base_dir / "code"
    data_dir = base_dir / "dataset"
    processed_path = base_dir / "processed" \
        / f"Dataloader_max_nodes_{MAX_NODES}_batch_{BATCH_SIZE}_test_{TEST_RATIO}.pt"

    # Use smaller batch size
    original_loader = get_filtered_dataloaders(
    root_dir=data_dir,
    processed_path=processed_path,
    batch_size=BATCH_SIZE,
    test_ratio=TEST_RATIO,
    max_nodes=MAX_NODES,
    )

    logger.info("Original dataset loaded with %d batches", len(original_loader))
    logger.info("Number of features: %d", original_loader.dataset.num_features)
    logger.info("Total number of nodes: %d",
                sum(graph.num_nodes for graph in original_loader.dataset))

    # Initialize model
    feature_dim = original_loader.dataset.num_features
    model = SampleGNN(
        input_dim=feature_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_GNN_LAYERS,  # Using max_distance + 2 layers
        dropout=DROPOUT,
        layer_norm=LAYER_NORM,
        residual_frequency=RESIDUAL_FREQUENCY
    )

    optimizer = AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    logger.info("Starting training with dynamic sampling...")
    train_sampled_gnn(
        model,
        original_loader,
        optimizer,
        epochs=EPOCHS,
        warmup_epochs=WARMUP_EPOCHS
    )

    # Load the best model
    best_model_path = 'best_sample_training_model.pth'
    model.load_state_dict(torch.load(best_model_path))
    logger.info("Best model loaded from %s", best_model_path)

    # Evaluate the best model on the entire dataset
    logger.info("Evaluating the best model on the entire dataset...")
    full_loader = get_filtered_dataloaders(
        root_dir=data_dir,
        processed_path=processed_path,
        batch_size=BATCH_SIZE,
        test_ratio=0.0,  # Use the entire dataset for evaluation
        max_nodes=MAX_NODES,
    )
    evaluate_sampled_model(model, full_loader, "Full")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, IOError) as e:
        logger.exception("Fatal error has occurred: %s", e)
