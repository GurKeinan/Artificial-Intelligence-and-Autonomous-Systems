import random
import torch
from torch_geometric.data import Data

def prune_graph_nodes(data):
    """
    Prune nodes from a graph based on random threshold selection.

    Args:
        data (Data): PyTorch Geometric Data object containing the graph

    Returns:
        Data: Pruned graph with subset of nodes
    """
    # Choose random threshold from [0.3, 0.5, 0.7]
    threshold = random.choice([0.3, 0.5, 0.7])

    total_nodes = data.num_nodes
    # Create mask for nodes to keep based on their indices
    keep_mask = torch.arange(total_nodes) / total_nodes < threshold

    # Get node indices to keep
    node_idx = torch.where(keep_mask)[0]

    # Update edge indices to only include edges between remaining nodes
    edge_mask = torch.isin(data.edge_index[0], node_idx) & torch.isin(data.edge_index[1], node_idx)
    new_edge_index = data.edge_index[:, edge_mask]

    # Create node index mapping for updating edge indices
    idx_map = {int(old_idx): new_idx for new_idx, old_idx in enumerate(node_idx)}
    new_edge_index = torch.tensor([[idx_map[int(i)] for i in new_edge_index[0]],
                                  [idx_map[int(i)] for i in new_edge_index[1]]])

    # Create new pruned graph
    pruned_data = Data(
        x=data.x[keep_mask],
        edge_index=new_edge_index,
        y=data.y[keep_mask],
    )

    return pruned_data
