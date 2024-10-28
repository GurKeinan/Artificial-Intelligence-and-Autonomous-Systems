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

from read_tree_search import get_filtered_dataloaders

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
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


def train_with_warmup(model, loader, optimizer, epochs, warmup_epochs=10,
                     accumulation_steps=4, max_grad_norm=1.0):
    """Training function with special handling for problematic batches"""

    def print_memory_stats(batch_idx=None):
        process = psutil.Process(os.getpid())
        ram_usage = process.memory_info().rss / 1024**2
        batch_info = f" (Batch {batch_idx})" if batch_idx is not None else ""
        logger.info(f"RAM Usage{batch_info}: {ram_usage:.2f}MB")

    def cleanup_memory():
        # Force garbage collection
        gc.collect()

        # Clear any remaining tensors not part of the model
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and not hasattr(obj, '_backward_hooks'):
                    del obj
            except Exception:
                pass

        gc.collect()

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        epochs=epochs,
        steps_per_epoch=len(loader) // accumulation_steps,
        pct_start=warmup_epochs/epochs
    )

    model = model.to(device)
    criterion = torch.nn.MSELoss()

    # Initial cleanup
    cleanup_memory()
    print_memory_stats()

    for epoch in range(epochs):
        logger.info(f"\nStarting Epoch {epoch + 1}")
        print_memory_stats()

        model.train()
        total_loss = 0
        valid_batches = 0
        optimizer.zero_grad()

        # Store intermediate tensors for cleanup
        stored_tensors = []

        for batch_idx, batch in tqdm(enumerate(loader), total=len(loader)):
            # Extra monitoring around problematic batch
            if batch_idx in range(115, 125):
                logger.info(f"\nProcessing sensitive batch {batch_idx}")
                print_memory_stats(batch_idx)
                cleanup_memory()

            try:
                batch = batch.to(device)
                predictions = model(batch)

                if hasattr(batch, 'train_mask'):
                    train_pred = predictions[batch.train_mask]
                    train_true = batch.y[batch.train_mask]

                    if len(train_pred) > 0:
                        loss = criterion(train_pred, train_true)
                        loss = loss / accumulation_steps
                        loss.backward()

                        # Store loss value and clear immediate tensors
                        total_loss += loss.item() * accumulation_steps
                        valid_batches += 1

                if (batch_idx + 1) % accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    cleanup_memory()

                # Immediate cleanup after each batch
                del batch, predictions
                if 'train_pred' in locals(): del train_pred
                if 'train_true' in locals(): del train_true
                if 'loss' in locals(): del loss
                cleanup_memory()

                # Extra monitoring after problematic batch
                if batch_idx in range(115, 125):
                    print_memory_stats(batch_idx)

            except RuntimeError as e:
                if "memory" in str(e).lower():
                    logger.error(f"Memory error on batch {batch_idx}. Attempting recovery...")
                    cleanup_memory()
                    print_memory_stats(batch_idx)

                    # Clear optimizer gradients
                    optimizer.zero_grad()

                    # Try to process with fresh memory
                    cleanup_memory()
                    continue
                else:
                    raise e

            # Additional cleanup every 50 batches
            if batch_idx % 50 == 0:
                cleanup_memory()
                print_memory_stats(batch_idx)

        if valid_batches > 0:
            avg_loss = total_loss / valid_batches
            logger.info(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}, '
                     f'LR: {scheduler.get_last_lr()[0]:.6f}')

        # Cleanup after epoch
        cleanup_memory()
        print_memory_stats()

        if epoch % 5 == 0:
            # Save model state before evaluation
            save_checkpoint(model, optimizer, epoch, f'checkpoint_epoch_{epoch}.pt')

            # Evaluate with memory cleanup
            cleanup_memory()
            evaluate(model, loader, "Train")
            cleanup_memory()
            evaluate(model, loader, "Test")
            cleanup_memory()

def evaluate(model, loader, mask_type="Test"):
    """Evaluation function with memory management"""
    model.eval()
    total_loss = 0
    num_batches = 0
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
                    total_loss += loss.item()
                    num_batches += 1

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

    if num_batches > 0:
        avg_loss = total_loss / num_batches
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
        warmup_epochs=10,
        accumulation_steps=4  # Accumulate gradients over 4 batches
    )

if __name__ == "__main__":
    main()