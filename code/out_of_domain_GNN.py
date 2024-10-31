

import logging
from pathlib import Path
from datetime import datetime
import shutil
import sys
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.optim.adamw import AdamW
from torch_geometric.nn import GATConv, GCNConv, Linear, GraphNorm, BatchNorm
from prepare_full_graph_dataset import get_filtered_dataloaders, FilteredTreeDataset, SerializableDataLoader, load_processed_data, save_processed_data

# Constants
MAX_NODES = 15000
HIDDEN_DIM = 256
NUM_LAYERS = 4
HEADS = 4
DROPOUT = 0.2
LAYER_NORM = True
RESIDUAL_FREQUENCY = 2
LR = 0.001
WEIGHT_DECAY = 0.01
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
log_filename = log_dir / f"out_of_domain_GNN_{timestamp}.log"

file_handler = logging.FileHandler(log_filename)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FullGraphsGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=4, heads=4, dropout=0.2, layer_norm=True, residual_frequency=2):
        super(FullGraphsGNN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual_frequency = residual_frequency

        self.input_proj = torch.nn.Sequential(
            Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            Linear(hidden_dim, hidden_dim)
        )

        self.gat_layers = torch.nn.ModuleList()
        self.gcn_layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.skip_layers = torch.nn.ModuleList()

        for _ in range(num_layers):
            self.gat_layers.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout))
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim, improved=True))
            self.norms.append(GraphNorm(hidden_dim) if layer_norm else BatchNorm(hidden_dim))
            self.skip_layers.append(Linear(hidden_dim, hidden_dim))

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
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_weight = self.edge_weight * torch.ones(edge_index.size(1), device=edge_index.device)
        x = self.input_proj(x)

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

        node_predictions = self.prediction_head(x)
        return torch.sigmoid(node_predictions).view(-1)

def train_with_warmup(model, loader, optimizer, epochs, warmup_epochs=10, max_grad_norm=1.0):
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
            optimizer.zero_grad()
            batch = batch.to(device)
            predictions = model(batch)
            loss = criterion(predictions, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            valid_batches += 1

        if valid_batches > 0:
            avg_loss = total_loss / valid_batches
            logger.info('Epoch %d, Loss: %.4f, LR: %.6f', epoch + 1, avg_loss, scheduler.get_last_lr()[0])

        if epoch % 5 == 0:
            # evaluate(model, loader, "Train")
            evaluate(model, loader)

def evaluate(model, loader):
    model.eval()
    total_loss = 0
    num_samples = 0
    criterion = torch.nn.MSELoss()

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predictions = model(batch)
            loss = criterion(predictions, batch.y)
            total_loss += loss.item() * len(batch.y)
            num_samples += len(batch.y)

            # if mask_type == "Train":
            #     mask = batch.train_mask
            # elif mask_type == "Test":
            #     mask = batch.test_mask
            # else:
            #     mask = torch.ones_like(batch.train_mask)

            # if mask.sum() > 0:
            #     loss = criterion(predictions[mask], batch.y[mask])
            #     total_loss += loss.item() * mask.sum()
            #     num_samples += mask.sum()

    if num_samples > 0:
        avg_loss = total_loss / num_samples
        logger.info('Evaluation average Loss: %.4f', avg_loss)
        return avg_loss
    return None

def filter_files_by_prefix(root_dir, prefix):
    root_path = Path(root_dir)
    filtered_files = []
    for dir_path in root_path.iterdir():
        if dir_path.is_dir() and dir_path.name.startswith(prefix):
            filtered_files.extend([str(p) for p in dir_path.rglob('*.pkl')])
    return filtered_files

def get_filtered_dataloaders_by_prefix(root_dir, processed_path, prefix, batch_size=32, test_ratio=0.2, max_nodes=1000):
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
        prefix="bw",
        batch_size=BATCH_SIZE,
        test_ratio=0.0,
        max_nodes=MAX_NODES
    )

    sp_dir = base_dir / "processed" / f"sp_dataloader_{MAX_NODES}.pt"
    eval_loader = get_filtered_dataloaders_by_prefix(
        root_dir=data_dir,
        processed_path=sp_dir,
        prefix="sp",
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
        warmup_epochs=WARMUP_EPOCHS
    )

    logger.info("Evaluating the model on the evaluation dataset...")
    evaluate(model, eval_loader)

if __name__ == "__main__":
    main()
