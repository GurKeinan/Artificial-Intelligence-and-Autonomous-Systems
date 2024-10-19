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

class ImprovedGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=4, heads=4, dropout=0.2):
        super(ImprovedGNN, self).__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()
        
        # Input projection
        self.input_proj = torch.nn.Linear(input_dim, hidden_dim)
        
        # Input layer
        self.convs.append(GATConv(hidden_dim, hidden_dim, heads=heads, concat=False))
        self.layer_norms.append(torch.nn.LayerNorm(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim, heads=heads, concat=False))
            self.layer_norms.append(torch.nn.LayerNorm(hidden_dim))
        
        # Output layer
        self.convs.append(GCNConv(hidden_dim, 1))
        
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Project input to hidden dimension
        x = self.input_proj(x)
        
        for i in range(self.num_layers - 1):
            identity = x
            x = self.convs[i](x, edge_index)
            x = self.layer_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual connection
            x = x + identity
        
        # Output layer
        x = self.convs[-1](x, edge_index)
        
        # No need for global pooling, we want node-level predictions
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
    num_samples = 0
    
    with torch.no_grad():
        for batch in loader:
            out = model(batch)
            mask = batch.test_mask if mask_type == "Test" else batch.train_mask
            loss = F.binary_cross_entropy(out[mask], batch.y[mask])
            mse = F.mse_loss(out[mask], batch.y[mask])
            total_loss += loss.item() * mask.sum().item()
            total_mse += mse.item() * mask.sum().item()
            num_samples += mask.sum().item()
    
    avg_loss = total_loss / num_samples
    rmse = mean_squared_error(batch.y[mask].cpu().numpy(), out[mask].cpu().numpy(), squared=False)
    
    print(f'{mask_type} Avg. Loss: {avg_loss:.4f}, RMSE: {rmse:.4f}')

def main():

    ### Load data
    base_dir = Path(__file__).resolve().parent
    if base_dir.name != "code":
        base_dir = base_dir / "code"
    data_dir = base_dir / "dataset"

    loader = get_dataloaders(root_dir=data_dir, batch_size=32, test_ratio=0.2)

    feature_dim = 10  # this is based on node_features in tree_to_graph
    model = ImprovedGNN(input_dim=feature_dim, hidden_dim=128, num_layers=6, heads=6, dropout=0.2)
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    loss_fn = torch.nn.MSELoss()
    epochs = 300

    train(model, loader, optimizer, loss_fn, epochs)
    print("Finished Training")

    test(model, loader, "Train")
    test(model, loader, "Test")

if __name__ == "__main__":
    main()