from pathlib import Path
from sklearn.metrics import mean_squared_error

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool

from read_tree_search import TreeDataset
from sliding_puzzle_A_star import SearchNode

class QuickResidualGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=3, dropout=0.2):
        super(QuickResidualGNN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # Input projection
        self.input_proj = torch.nn.Linear(input_dim, hidden_dim)

        # Graph Convolutional layers
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Output layer
        self.output = GCNConv(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Project input to hidden dimension
        x = self.input_proj(x)

        # Graph Convolutional layers with residual connections
        for conv in self.convs:
            residual = x
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual  # Residual connection

        # Output layer
        x = self.output(x, edge_index)

        return torch.sigmoid(x).view(-1)

def get_dataloaders(root_dir, batch_size=32, test_ratio=0.2):
    dataset = TreeDataset(root_dir=root_dir, test_ratio=test_ratio)
    print(f"Dataset size: {len(dataset)}")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

def train(model, loader, optimizer, loss_fn, epochs):
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = loss_fn(out[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss:.4f}')

def test(model, loader, mask_type):
    model.eval()
    total_loss = 0
    total_mse = 0
    total_sse = 0
    num_samples = 0
    num_trees = 0

    preds = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            out = model(batch)

            if mask_type == "Full":
                mask = batch.test_mask | batch.train_mask
            elif mask_type == "Train":
                mask = batch.train_mask
            else:
                mask = batch.test_mask

            loss = F.binary_cross_entropy(out[mask], batch.y[mask])
            mse = F.mse_loss(out[mask], batch.y[mask])
            sse = (out[mask] - batch.y[mask]).pow(2).sum()

            total_loss += loss.item() * mask.sum().item()
            total_mse += mse.item()
            total_sse += sse.item()
            num_samples += mask.sum().item()
            num_trees += batch.num_graphs

    avg_loss = total_loss / num_samples
    # rmse = (total_mse ** 0.5) / num_trees

    mse = total_sse / num_samples
    # rmse = (score ** 0.5) / num_trees

    print(f'{mask_type} Avg. Loss: {avg_loss:.4f}, MSE: {mse:.4f}')

def main():

    ### Load data
    base_dir = Path(__file__).resolve().parent
    if base_dir.name != "code":
        base_dir = base_dir / "code"
    data_dir = base_dir / "dataset"

    loader = get_dataloaders(root_dir=data_dir, batch_size=32, test_ratio=0.2)

    feature_dim = 10  # this is based on node_features in tree_to_graph
    model = QuickResidualGNN(input_dim=feature_dim, hidden_dim=64, num_layers=3, dropout=0.2)
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    loss_fn = torch.nn.MSELoss()
    epochs = 200

    train(model, loader, optimizer, loss_fn, epochs)
    print("Finished Training")

    test(model, loader, "Train")
    test(model, loader, "Test")
    test(model, loader, "Full")

if __name__ == "__main__":
    main()
