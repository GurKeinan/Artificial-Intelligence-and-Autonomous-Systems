import torch
from torch_geometric.loader import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from read_tree_search import TreeDataset
from sliding_puzzle_A_star import SearchNode

def get_dataloaders(root_dir, batch_size=8, test_ratio=0.2):
    # Load
    dataset = TreeDataset(root_dir=root_dir, test_ratio=test_ratio)
    print(f"Dataset size: {len(dataset)}")
    # Create dataloaders:
    loader = DataLoader(dataset, batch_size, shuffle=True)
    return loader


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)  # Output is node-level regression predictions
        return x

def train(model, loader, optimizer, loss_fn, epochs):

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_correct = 0

        for batch in loader:
            optimizer.zero_grad()
            # Forward pass
            out = model(batch)  # Predicts regression values for all nodes
            # Apply the train mask to select only the nodes used for training
            loss = loss_fn(out[batch.train_mask].squeeze(), batch.y[batch.train_mask])
            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pred = out[batch.train_mask].squeeze().round()
            epoch_correct += pred.eq(batch.y[batch.train_mask]).sum().item()
        
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss}, Accuracy: {epoch_correct / len(loader.dataset)}')

def test(model, loader, mask_type):
    model.eval()
    total_loss = 0
    correct = 0
    if mask_type == "Test":
        mask = loader.dataset.test_mask
    else:
        mask = loader.dataset.train_mask

    for batch in loader:
        with torch.no_grad():
            out = model(batch)
            loss = F.mse_loss(out[mask].squeeze(), batch.y[mask])
            total_loss += loss.item()

            pred = out[mask].squeeze().round()
            correct += pred.eq(batch.y[mask]).sum().item()

    print(f'{mask_type} Loss: {total_loss}')
    print(f'{mask_type} Accuracy: {correct / len(loader.dataset)}')

def main():
    loader = get_dataloaders("code/puzzle_tree_dataset/", batch_size=4, test_ratio=0.2)

    feature_dim = 10 # this is based on node_features in tree_to_graph
    model = GCN(input_dim=feature_dim, hidden_dim=64, output_dim=1)
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    loss_fn = torch.nn.MSELoss()
    epochs = 100

    train(model, loader, optimizer, loss_fn, epochs)
    print("Finished Training")

    test(model, loader, "Train")
    test(model, loader, "Test")
    

if __name__ == "__main__":
    main()