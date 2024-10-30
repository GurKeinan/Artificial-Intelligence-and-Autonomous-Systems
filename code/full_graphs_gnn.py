"""
full graphs gnn Training Script

This script trains a Graph Neural Network (GNN) model using a combination of
Graph Attention Networks (GAT) and Graph Convolutional Networks (GCN).
The model is designed to handle large graphs with up to 20,000 nodes and includes
various features such as layer normalization, residual connections, and dropout for regularization.

Modules:
    - logging: For logging training progress and results.
    - torch: PyTorch library for tensor operations and neural networks.
    - pathlib: For handling file paths.
    - tqdm: For displaying progress bars.
    - datetime: For timestamping log files.
    - torch.nn.functional: For various neural network functions.
    - torch_geometric.nn: For graph neural network layers and utilities.
    - read_tree_search: Custom module for loading data.

Constants:
    - MAX_NODES: Maximum number of nodes in a graph.
    - HIDDEN_DIM: Dimension of hidden layers.
    - NUM_LAYERS: Number of GNN layers.
    - HEADS: Number of attention heads in GAT layers.
    - DROPOUT: Dropout rate for regularization.
    - LAYER_NORM: Boolean flag for using layer normalization.
    - RESIDUAL_FREQUENCY: Frequency of residual connections.
    - LR: Learning rate for the optimizer.
    - WEIGHT_DECAY: Weight decay for the optimizer.
    - EPOCHS: Number of training epochs.
    - WARMUP_EPOCHS: Number of warmup epochs for learning rate scheduling.
    - BATCH_SIZE: Batch size for training.
    - TEST_RATIO: Ratio of test data.

Classes:
    - FullGraphsGNN: Defines the GNN model architecture.

Functions:
    - train_with_warmup: Trains the model with learning rate warmup and gradient clipping.
    - evaluate: Evaluates the model on a given dataset.
    - save_checkpoint: Saves the model checkpoint.
    - main: Main function to set up data, initialize the model, and start training.

Usage:
    Run the script directly to start training the GNN model.
"""

import logging
from pathlib import Path
from datetime import datetime
import sys
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.optim.adamw import AdamW
from torch_geometric.nn import GATConv, GCNConv, Linear
from torch_geometric.nn import GraphNorm, BatchNorm


from prepare_full_graph_dataset import get_filtered_dataloaders

# filtering constants
MAX_NODES = 15000
#model constants
HIDDEN_DIM = 256
NUM_LAYERS = 4
HEADS = 4
DROPOUT = 0.2
LAYER_NORM = True
RESIDUAL_FREQUENCY = 2
# optimizer constants
LR = 0.001
WEIGHT_DECAY = 0.01
# training constants
EPOCHS = 100
WARMUP_EPOCHS = 10
BATCH_SIZE = 16
TEST_RATIO = 0.2

repo_root = Path(__file__).resolve().parent
dataset_creation_path = repo_root / "dataset_creation"
sys.path.append(str(dataset_creation_path))

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Create timestamp for the log file
timestamp = datetime.now().strftime('%d.%m.%Y_%H:%M:%S')
log_filename = log_dir / f"training_GNN_{timestamp}.log"

file_handler = logging.FileHandler(log_filename)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FullGraphsGNN(torch.nn.Module):
    """
    FullGraphsGNN is a graph neural network model that combines Graph Attention Networks (GAT)
    and Graph Convolutional Networks (GCN) for multi-scale feature extraction and prediction.
    Args:
        input_dim (int): Dimension of the input features.
        hidden_dim (int, optional): Dimension of the hidden layers. Default is 256.
        num_layers (int, optional): Number of GAT and GCN layers. Default is 4.
        heads (int, optional): Number of attention heads in GAT layers. Default is 4.
        dropout (float, optional): Dropout rate for regularization. Default is 0.2.
        layer_norm (bool, optional): Whether to use layer normalization. Default is True.
        residual_frequency (int, optional): Frequency of residual connections. Default is 2.
    Attributes:
        num_layers (int): Number of GAT and GCN layers.
        dropout (float): Dropout rate for regularization.
        residual_frequency (int): Frequency of residual connections.
        input_proj (torch.nn.Sequential): Input projection layer.
        gat_layers (torch.nn.ModuleList): List of GAT layers.
        gcn_layers (torch.nn.ModuleList): List of GCN layers.
        norms (torch.nn.ModuleList): List of normalization layers.
        skip_layers (torch.nn.ModuleList): List of skip connection layers.
        prediction_head (torch.nn.Sequential): Prediction head for final output.
        edge_weight (torch.nn.Parameter): Edge weight parameter.
    Methods:
        forward(data):
            Forward pass of the model.
            Args:
                data (torch_geometric.data.Data): Input data containing node features,
                edge indices, and batch information.
            Returns:
                torch.Tensor: Sigmoid-activated node predictions.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        num_layers=4,
        heads=4,
        dropout=0.2,
        layer_norm=True,
        residual_frequency=2
    ):
        super(FullGraphsGNN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual_frequency = residual_frequency

        # Input projection with larger capacity
        self.input_proj = torch.nn.Sequential(
            Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            Linear(hidden_dim, hidden_dim)
        )

        # Multiple types of graph convolution layers
        self.gat_layers = torch.nn.ModuleList()
        self.gcn_layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.skip_layers = torch.nn.ModuleList()

        for _ in range(num_layers):
            # GAT layer for capturing important node relationships
            self.gat_layers.append(GATConv(
                hidden_dim,
                hidden_dim // heads,
                heads=heads,
                dropout=dropout
            ))

            # GCN layer for neighborhood aggregation
            self.gcn_layers.append(GCNConv(
                hidden_dim,
                hidden_dim,
                improved=True
            ))

            # Normalization layers
            if layer_norm:
                self.norms.append(GraphNorm(hidden_dim))
            else:
                self.norms.append(BatchNorm(hidden_dim))

            # Skip connection layers
            self.skip_layers.append(Linear(hidden_dim, hidden_dim))

        # Prediction head with multiple components
        self.prediction_head = torch.nn.Sequential(
            Linear(hidden_dim, hidden_dim),  # Changed from hidden_dim * 2
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            Linear(hidden_dim // 2, 1)
        )

        # Edge weight parameter
        self.edge_weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, data):
        """
        Forward pass for the GNN model.
        Args:
            data (torch_geometric.data.Data): Input data containing node features `x`,
            edge indices `edge_index`, and batch indices `batch`.
        Returns:
            torch.Tensor: Node-level predictions after applying the GNN model,
            with values in the range [0, 1].
        """

        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Make graph bidirectional and weight edges
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_weight = self.edge_weight * torch.ones(edge_index.size(1),
                                                  device=edge_index.device)

        # Initial feature projection
        x = self.input_proj(x)

        # Multi-scale feature extraction
        for i in range(self.num_layers):
            # Store previous representation for residual
            prev_x = x

            # GAT for attention-based message passing
            gat_out = self.gat_layers[i](x, edge_index)

            # GCN for structural feature extraction
            gcn_out = self.gcn_layers[i](x, edge_index, edge_weight)

            # Combine GAT and GCN features
            x = gat_out + gcn_out

            # Apply normalization
            x = self.norms[i](x, batch)

            # Non-linearity and dropout
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # Residual connection with learned skip
            if i % self.residual_frequency == 0:
                skip = self.skip_layers[i](prev_x)
                x = x + skip

        node_predictions = self.prediction_head(x)

        return torch.sigmoid(node_predictions).view(-1)

def train_with_warmup(model, loader, optimizer, epochs, warmup_epochs=10, max_grad_norm=1.0):
    """
    Train a model with a learning rate warmup phase.
    Args:
        model (torch.nn.Module): The neural network model to be trained.
        loader (torch.utils.data.DataLoader): DataLoader providing the training data.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        epochs (int): Total number of training epochs.
        warmup_epochs (int, optional): Number of epochs for learning rate warmup. Default is 10.
        max_grad_norm (float, optional): Maximum norm for gradient clipping. Default is 1.0.
    Returns:
        None
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

    for epoch in range(epochs):
        logger.info("Starting Epoch %d", epoch + 1)

        model.train()
        total_loss = 0
        valid_batches = 0

        for batch in tqdm(loader, total=len(loader)):
            optimizer.zero_grad()  # Zero gradients at start of each batch

            batch = batch.to(device)
            predictions = model(batch)

            if hasattr(batch, 'train_mask'):
                train_pred = predictions[batch.train_mask]
                train_true = batch.y[batch.train_mask]

                if len(train_pred) > 0:
                    loss = criterion(train_pred, train_true)
                    loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                    # Immediate optimizer step
                    optimizer.step()
                    scheduler.step()

                    # Store loss value
                    total_loss += loss.item()
                    valid_batches += 1

        if valid_batches > 0:
            avg_loss = total_loss / valid_batches
            logger.info('Epoch %d, Loss: %.4f, LR: %.6f', epoch + 1,
                        avg_loss, scheduler.get_last_lr()[0])

        if epoch % 5 == 0:
            evaluate(model, loader, "Train")
            evaluate(model, loader, "Test")

def evaluate(model, loader, mask_type="Test"):
    """
    Evaluate the performance of a model on a given dataset.
    Args:
        model (torch.nn.Module): The model to evaluate.
        loader (torch.utils.data.DataLoader): DataLoader providing the dataset.
        mask_type (str, optional): Type of mask to use for evaluation.
                                   Options are "Train", "Test", or "Full".
                                   Defaults to "Test".
    Returns:
        float or None: The average loss over the evaluated samples if any samples
                       are evaluated, otherwise None.
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

def main():
    """
    Main function to set up and train a FullGraphsGNN model.
    This function performs the following steps:
    1. Sets a random seed for reproducibility.
    2. Logs the device being used.
    3. Loads the dataset with a specified batch size and filtering parameters.
    4. Logs the number of batches and total samples in the dataset.
    5. Initializes the FullGraphsGNN model with specified hyperparameters.
    6. Initializes the optimizer with weight decay.
    7. Trains the model using warmup and gradient accumulation.
    Note:
        The function relies on several global variables and functions:
        - `torch`: PyTorch library for tensor operations.
        - `logger`: Logger for logging information.
        - `device`: Device to be used for computation (e.g., 'cpu' or 'cuda').
        - `Path`: Path class from the pathlib module for handling file paths.
        - `MAX_NODES`, `BATCH_SIZE`, `TEST_RATIO`, `HIDDEN_DIM`, `NUM_LAYERS`,
          `HEADS`, `DROPOUT`, `LAYER_NORM`, `RESIDUAL_FREQUENCY`, `LR`,
          `WEIGHT_DECAY`, `EPOCHS`, `WARMUP_EPOCHS`: Hyperparameters for the model and training.
        - `get_filtered_dataloaders`: Function to load and filter the dataset.
        - `FullGraphsGNN`: Class representing the GNN model.
        - `train_with_warmup`: Function to train the model with warmup and gradient accumulation.
    """

    # Set random seed for reproducibility
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
    loader = get_filtered_dataloaders(
    root_dir=data_dir,
    processed_path=processed_path,
    batch_size=BATCH_SIZE,
    test_ratio=TEST_RATIO,
    max_nodes=MAX_NODES,    # Added filtering parameters
)
    logger.info("Data loaded with %d batches", len(loader))
    logger.info("Total number of graphs: %d", len(loader.dataset))
    logger.info("Number of features: %d", loader.dataset.num_features)
    logger.info("Total number of nodes: %d", sum(graph.num_nodes for graph in loader.dataset))

    # Initialize model
    feature_dim = loader.dataset.num_features
    model = FullGraphsGNN(
        input_dim=feature_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        heads=HEADS,
        dropout=DROPOUT,
        layer_norm=LAYER_NORM,
        residual_frequency=RESIDUAL_FREQUENCY
    )

    # Initialize optimizer with weight decay
    optimizer = AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    # Train with warmup and gradient accumulation
    logger.info("Starting training...")
    train_with_warmup(
        model,
        loader,
        optimizer,
        epochs=EPOCHS,
        warmup_epochs=WARMUP_EPOCHS
    )

if __name__ == "__main__":
    main()
