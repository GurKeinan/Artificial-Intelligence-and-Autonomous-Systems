from pathlib import Path
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, Linear, global_mean_pool, global_add_pool
from torch_geometric.nn import GraphNorm, BatchNorm
from pathlib import Path
from torch_geometric.loader import DataLoader
from read_tree_search import TreeDataset

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')


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

def get_dataloaders(root_dir, batch_size=32, test_ratio=0.2):
    dataset = TreeDataset(root_dir=root_dir, test_ratio=test_ratio)
    print(f"Dataset size: {len(dataset)}")
    # Don't shuffle during training to maintain mask correspondence
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return train_loader

def train_with_warmup(model, loader, optimizer, epochs, warmup_epochs=10):
    """Training function with learning rate warmup and gradient clipping"""
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        epochs=epochs,
        steps_per_epoch=len(loader),
        pct_start=warmup_epochs/epochs
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = torch.nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(batch)
            
            # Calculate loss only on training nodes
            if hasattr(batch, 'train_mask'):
                # Use boolean indexing
                train_pred = predictions[batch.train_mask]
                train_true = batch.y[batch.train_mask]
                
                if len(train_pred) > 0:  # Only compute loss if we have training nodes
                    loss = criterion(train_pred, train_true)
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    scheduler.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
            
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')

        if epoch % 5 == 0:
            evaluate(model, loader, "Train")
            evaluate(model, loader, "Test")

def evaluate(model, loader, mask_type="Test"):
    model.eval()
    total_loss = 0
    num_batches = 0
    criterion = torch.nn.MSELoss()
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predictions = model(batch)
            
            # Select appropriate mask
            if mask_type == "Train":
                mask = batch.train_mask
            elif mask_type == "Test":
                mask = batch.test_mask
            else:  # "Full"
                mask = torch.ones_like(batch.train_mask)
            
            if mask.sum() > 0:  # Only compute loss if we have nodes for this mask type
                loss = criterion(predictions[mask], batch.y[mask])
                total_loss += loss.item()
                num_batches += 1
    
    if num_batches > 0:
        avg_loss = total_loss / num_batches
        print(f'{mask_type} Average Loss: {avg_loss:.4f}')
        return avg_loss
    return None

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    print(f"Using device: {device}")
    
    # Load data
    base_dir = Path(__file__).resolve().parent
    if base_dir.name != "code":
        base_dir = base_dir / "code"
    data_dir = base_dir / "dataset"
    
    # Smaller batch size to handle larger model
    loader = get_dataloaders(root_dir=data_dir, batch_size=16, test_ratio=0.2)
    
    # Initialize model with improved architecture
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
    
    # Train with warmup
    print("Starting training...")
    train_with_warmup(model, loader, optimizer, epochs=100, warmup_epochs=10)
    
    # Evaluate
    print("\nEvaluating model...")
    evaluate(model, loader, "Train")
    evaluate(model, loader, "Test")
    evaluate(model, loader, "Full")

if __name__ == "__main__":
    main()
