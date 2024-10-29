import logging
from pathlib import Path
from datetime import datetime
import sys
import test
import torch
import torch.nn.functional as F
from torch.optim.adamw import AdamW
from torch_geometric.nn import GATConv, GCNConv, Linear, GraphNorm, BatchNorm
from torch_geometric.data import Data, InMemoryDataset, DataLoader, Batch
from torch_geometric.utils import k_hop_subgraph
from tqdm import tqdm
import random
from prepare_full_graph_dataset import get_filtered_dataloaders

# constants for filtering
MAX_NODES = 10000  # Maximum number of nodes in a graph

# Constants for sampling
SAMPLES_PER_EPOCH = 10000  # Total samples to generate per epoch
MAX_DISTANCE = 3  # Maximum hop distance for subgraphs
NUM_GNN_LAYERS = MAX_DISTANCE + 2  # Number of GNN layers based on max distance

# Model constants
HIDDEN_DIM = 256

HEADS = 4
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
    """DataLoader that dynamically samples subgraphs during iteration"""
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
        """Sample a single subgraph from a given graph"""
        num_nodes = graph.x.size(0)
        node_idx = random.randrange(num_nodes)

        # Get k-hop subgraph
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
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

class SampledGNN(torch.nn.Module):
    """Modified GNN to handle sampled subgraphs"""
    def __init__(self, input_dim, hidden_dim, num_layers, heads,
                 dropout, layer_norm, residual_frequency):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual_frequency = residual_frequency

        # Input projection
        self.input_proj = torch.nn.Sequential(
            Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            Linear(hidden_dim, hidden_dim)
        )

        # GNN layers
        self.gat_layers = torch.nn.ModuleList()
        self.gcn_layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.skip_layers = torch.nn.ModuleList()

        for _ in range(num_layers):
            self.gat_layers.append(GATConv(
                hidden_dim,
                hidden_dim // heads,
                heads=heads,
                dropout=dropout
            ))

            self.gcn_layers.append(GCNConv(
                hidden_dim,
                hidden_dim,
                improved=True
            ))

            self.norms.append(GraphNorm(hidden_dim) if layer_norm else BatchNorm(hidden_dim))
            self.skip_layers.append(Linear(hidden_dim, hidden_dim))

        # Prediction head
        self.prediction_head = torch.nn.Sequential(
            Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            Linear(hidden_dim // 2, 1)
        )

        self.edge_weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Make graph bidirectional and weight edges
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_weight = self.edge_weight * torch.ones(edge_index.size(1), device=edge_index.device)

        # Initial feature projection
        x = self.input_proj(x)

        # Multi-scale feature extraction
        for i in range(self.num_layers):
            prev_x = x

            gat_out = self.gat_layers[i](x, edge_index)
            gcn_out = self.gcn_layers[i](x, edge_index, edge_weight)

            x = gat_out + gcn_out
            x = self.norms[i](x, batch)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            if i % self.residual_frequency == 0:
                skip = self.skip_layers[i](prev_x)
                x = x + skip

        # Get predictions
        node_predictions = self.prediction_head(x)

        return torch.sigmoid(node_predictions).view(-1)

def train_sampled_gnn(model, original_loader, optimizer, epochs,
                     warmup_epochs=10, max_grad_norm=1.0):
    """Train the GNN using dynamically sampled subgraphs"""
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        epochs=epochs,
        steps_per_epoch=len(original_loader),
        pct_start=warmup_epochs/epochs
    )

    model = model.to(device)
    criterion = torch.nn.MSELoss()

    # Create sampled loader
    sampled_loader = DynamicSampledLoader(
        original_dataset=original_loader.dataset,
        samples_per_epoch=SAMPLES_PER_EPOCH,
        max_distance=MAX_DISTANCE,
        batch_size=BATCH_SIZE
    )

    for epoch in range(epochs):
        logger.info(f"Starting Epoch {epoch + 1}")
        model.train()
        total_loss = 0
        valid_batches = 0

        for batch in tqdm(sampled_loader, total=len(sampled_loader)):
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

        if valid_batches > 0:
            avg_loss = total_loss / valid_batches
            logger.info(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')

        if epoch % 5 == 0:
            evaluate_sampled_model(model, sampled_loader, "Train")
            evaluate_sampled_model(model, sampled_loader, "Test")

def evaluate_sampled_model(model, loader, mask_type="Test"):
    """Evaluate the model on sampled subgraphs"""
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
        logger.info(f'{mask_type} Average Loss: {avg_loss:.4f}')
        return avg_loss
    return None

def main():
    torch.manual_seed(42)
    logger.info(f"Using device: {device}")

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

    logger.info(f"Original dataset loaded with {len(original_loader)} batches")
    logger.info(f"Number of features: {original_loader.dataset.num_features}")

    # Initialize model
    feature_dim = original_loader.dataset.num_features
    model = SampledGNN(
        input_dim=feature_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_GNN_LAYERS,  # Using max_distance + 2 layers
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

    logger.info("Starting training with dynamic sampling...")
    train_sampled_gnn(
        model,
        original_loader,
        optimizer,
        epochs=EPOCHS,
        warmup_epochs=WARMUP_EPOCHS
    )

if __name__ == "__main__":
    main()