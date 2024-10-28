from pathlib import Path
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, Linear, global_mean_pool, global_add_pool
from torch_geometric.nn import GraphNorm, BatchNorm
from torch_geometric.loader import DataLoader
import logging
from tqdm import tqdm
import gc
import psutil
import os
from datetime import datetime

from read_tree_search import get_filtered_dataloaders

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

class ImprovedSearchGNN(torch.nn.Module):
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
        super(ImprovedSearchGNN, self).__init__()
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

        for i in range(num_layers):
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
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Make graph bidirectional and weight edges
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_weight = self.edge_weight * torch.ones(edge_index.size(1),
                                                  device=edge_index.device)

        # Initial feature projection
        x = self.input_proj(x)
        original_features = x

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
    """Training function with simplified batch processing"""

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        epochs=epochs,
        steps_per_epoch=len(loader),  # No more division by accumulation_steps
        pct_start=warmup_epochs/epochs
    )

    model = model.to(device)
    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):
        logger.info(f"Starting Epoch {epoch + 1}")

        model.train()
        total_loss = 0
        valid_batches = 0

        for batch_idx, batch in tqdm(enumerate(loader), total=len(loader)):
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
            logger.info(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}, '
                     f'LR: {scheduler.get_last_lr()[0]:.6f}')

        if epoch % 5 == 0:
            # Save model state before evaluation
            save_checkpoint(model, optimizer, epoch, f'checkpoint_epoch_{epoch}.pt')
            evaluate(model, loader, "Train")
            evaluate(model, loader, "Test")

def evaluate(model, loader, mask_type="Test"):
    """Evaluation function with memory management"""
    model.eval()
    total_loss = 0
    num_samples = 0
    criterion = torch.nn.MSELoss()

    with torch.no_grad():
        for batch in loader:
            try:
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

                # Cleanup
                del batch, predictions, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    logger.error(f"OOM during evaluation. Skipping batch...")
                    continue
                else:
                    raise e

    if num_samples > 0:
        avg_loss = total_loss / num_samples
        logger.info(f'{mask_type} Average Loss: {avg_loss:.4f}')
        return avg_loss
    return None

def save_checkpoint(model, optimizer, epoch, filename):
    """Save training checkpoint"""
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    torch.save(checkpoint, checkpoint_dir / filename)
    logger.info(f"Saved checkpoint for epoch {epoch}")

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    logger.info(f"Using device: {device}")

    # Load data with smaller batch size
    base_dir = Path(__file__).resolve().parent
    if base_dir.name != "code":
        base_dir = base_dir / "code"
    data_dir = base_dir / "dataset"
    processed_path = base_dir / "processed" / "dataloader.pt"

    # Use smaller batch size
    loader = get_filtered_dataloaders(
    root_dir=data_dir,
    processed_path=processed_path,
    batch_size=16,
    test_ratio=0.2,
    max_nodes=10000,    # Added filtering parameters
    max_depth=16,
    max_branching=10
)
    logger.info(f"Data loaded with {len(loader)} batches")

    # Initialize model
    feature_dim = 10
    model = ImprovedSearchGNN(
        input_dim=feature_dim,
        hidden_dim=256,
        num_layers=4,
        heads=4,
        dropout=0.2,
        layer_norm=True,
        residual_frequency=2
    )

    # Initialize optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )

    # Train with warmup and gradient accumulation
    logger.info("Starting training...")
    train_with_warmup(
        model,
        loader,
        optimizer,
        epochs=100,
        warmup_epochs=10
    )

if __name__ == "__main__":
    main()