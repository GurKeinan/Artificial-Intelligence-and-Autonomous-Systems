""" Graph Neural Network models for search progress prediction. """

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.nn import GraphNorm, BatchNorm


class HeavyGNN(torch.nn.Module):
    """
    HeavyGNN is a graph neural network model that combines Graph Attention Networks (GAT)
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
        super(HeavyGNN, self).__init__()
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
            torch.Tensor: Node-level predictions after sigmoid activation.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Ensure edge_index is long type and on correct device
        edge_index = edge_index.long()

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

            # Ensure inputs are on same device
            x = x.to(edge_index.device)

            # GAT for attention-based message passing
            gat_out = self.gat_layers[i](x, edge_index)

            # GCN for structural feature extraction
            gcn_out = self.gcn_layers[i](x, edge_index, edge_weight)

            # Rest of the layer processing remains same
            x = gat_out + gcn_out
            x = self.norms[i](x, batch)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            if i % self.residual_frequency == 0:
                skip = self.skip_layers[i](prev_x)
                x = x + skip

        node_predictions = self.prediction_head(x)
        return torch.sigmoid(node_predictions).view(-1)


class LightGNN(torch.nn.Module):
    """
    A Graph Neural Network (GNN) model with sampling and residual connections.
    Args:
        input_dim (int): Dimension of the input features.
        hidden_dim (int): Dimension of the hidden layers.
        num_layers (int): Number of GNN layers.
        dropout (float): Dropout rate.
        layer_norm (bool): Whether to use layer normalization.
        residual_frequency (int): Frequency of residual connections.
    Attributes:
        num_layers (int): Number of GNN layers.
        dropout (float): Dropout rate.
        residual_frequency (int): Frequency of residual connections.
        input_proj (torch.nn.Sequential): Input projection layer.
        gcn_layers (torch.nn.ModuleList): List of GCN layers.
        norms (torch.nn.ModuleList): List of normalization layers.
        skip_layers (torch.nn.ModuleList): List of skip connection layers.
        prediction_head (torch.nn.Sequential): Prediction head for node classification.
        edge_weight (torch.nn.Parameter): Edge weight parameter.
    Methods:
        forward(data):
            Forward pass of the GNN model.
            Args:
                data (torch_geometric.data.Data): Input data containing node features,
                edge indices, and batch information.
            Returns:
                torch.Tensor: Node predictions after applying the GNN model.
    """

    def __init__(self, input_dim, hidden_dim, num_layers, dropout, layer_norm, residual_frequency):
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
        self.gcn_layers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.skip_layers = torch.nn.ModuleList()

        for _ in range(num_layers):
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
        """
        Perform a forward pass through the GNN model.
        Args:
            data (torch_geometric.data.Data): Input data containing node features,
            edge indices, and batch information.
        Returns:
            torch.Tensor: The predicted node labels as a 1D tensor with values in the range [0, 1].
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Ensure edge_index is long type and on correct device
        edge_index = edge_index.long()

        # Make graph bidirectional and weight edges
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_weight = self.edge_weight * torch.ones(edge_index.size(1),
                                                  device=edge_index.device)

        # Initial feature projection
        x = self.input_proj(x)

        # Multi-scale feature extraction
        for i in range(self.num_layers):
            prev_x = x

            # Ensure inputs are on same device
            x = x.to(edge_index.device)

            gcn_out = self.gcn_layers[i](x, edge_index, edge_weight)

            x = gcn_out
            x = self.norms[i](x, batch)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            if i % self.residual_frequency == 0:
                skip = self.skip_layers[i](prev_x)
                x = x + skip

        # Get predictions
        node_predictions = self.prediction_head(x)

        return torch.sigmoid(node_predictions).view(-1)