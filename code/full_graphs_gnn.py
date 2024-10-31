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
from xgboost import train


from prepare_full_graph_dataset import get_filtered_dataloaders, SerializableDataLoader

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

def train_with_warmup(
    model,
    train_loader,
    test_loader,
    optimizer,
    epochs,
    warmup_epochs=10,
    max_grad_norm=1.0,
    patience=10,
    eval_every=5
):
    """
    Train model with warmup using separate train/test loaders and early stopping.
    """
    model = model.to(device)
    criterion = torch.nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=warmup_epochs/epochs
    )

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            batch = batch.to(device)
            optimizer.zero_grad()

            predictions = model(batch)
            loss = criterion(predictions, batch.y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)

        # Validation phase
        if epoch % eval_every == 0:
            val_loss = evaluate(model, test_loader, "Test")

            logger.info(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, '
                       f'Val Loss = {val_loss:.4f}, '
                       f'LR = {scheduler.get_last_lr()[0]:.6f}')

            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), 'best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f'Early stopping triggered after {epoch+1} epochs')
                    break
        else:
            logger.info(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, '
                       f'LR = {scheduler.get_last_lr()[0]:.6f}')



def evaluate(model, loader, mask_type):

    model.eval()
    total_loss = 0
    num_samples = 0
    criterion = torch.nn.MSELoss()

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predictions = model(batch)
            loss = criterion(predictions, batch.y)
            total_loss += loss.item() * len(batch.y) # get sse (sum of squared errors) for the batch
            num_samples += len(batch.y)

    avg_loss = total_loss / num_samples # calculate mse as the sse (sum over all samples) divided by the number of samples
    logger.info('%s Average Loss: %.4f', mask_type, avg_loss)
    return avg_loss

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
    full_loader = get_filtered_dataloaders(
    root_dir=data_dir,
    processed_path=processed_path,
    batch_size=BATCH_SIZE,
    test_ratio=TEST_RATIO,
    max_nodes=MAX_NODES,    # Added filtering parameters
    )

    # Get the dataset from the full loader
    dataset = full_loader.dataset

    # Calculate split sizes
    train_size = int((1 - TEST_RATIO) * len(dataset))
    test_size = len(dataset) - train_size

    # Split dataset
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size],generator=torch.Generator().manual_seed(42))

    # Create separate loaders
    train_loader = SerializableDataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True
                )

    test_loader = SerializableDataLoader(
                test_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False
                )
    feature_dim = full_loader.dataset.num_features

    # logs
    logger.info("Training dataset loaded with %d batches", len(train_loader))
    logger.info("Evaluation dataset loaded with %d batches", len(test_loader))
    logger.info("Feature dimension: %d", feature_dim)
    logger.info("Total nodes in full dataset: %d", sum([data.num_nodes for data in full_loader.dataset]))
    logger.info("Total nodes in training dataset: %d", sum([data.num_nodes for data in train_loader.dataset]))
    logger.info("Total nodes in evaluation dataset: %d", sum([data.num_nodes for data in test_loader.dataset]))

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
        train_loader,
        test_loader,
        optimizer,
        epochs=EPOCHS,
        warmup_epochs=WARMUP_EPOCHS
    )

if __name__ == "__main__":
    main()
