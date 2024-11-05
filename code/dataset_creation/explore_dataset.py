"""
This script explores the dataset and counts the number of nodes in each file.
It prints the summary statistics.
"""

import pickle
from pathlib import Path
import sys

def nodes_num(root):
    """Count the number of nodes in a search tree starting from the root node."""
    num_nodes = 0

    def traverse(node):
        """Recursively traverse the tree and count nodes."""
        nonlocal num_nodes
        num_nodes += 1  # Increment the count for the current node
        for child in node.children:
            traverse(child)  # Traverse child nodes

    traverse(root)  # Start traversal from the root
    return num_nodes  # Return the total number of nodes

# Set up paths to dataset directories
repo_dir = Path(__file__).resolve().parent.parent
dataset_dir = repo_dir / "dataset"

# Lists to store directories starting with specific prefixes
dirs_start_with_bw = []
dirs_start_with_sp = []

# Populate directories based on their prefixes
for dir in dataset_dir.iterdir():
    if dir.is_dir():
        if dir.name.startswith("bw"):
            dirs_start_with_bw.append(dir)
        elif dir.name.startswith("sp"):
            dirs_start_with_sp.append(dir)

# Calculate node numbers for block world files
bw_nodes_num_list = []
bw_files_num = 0
for dir in dirs_start_with_bw:
    for file in dir.iterdir():
        try:
            with open(file, "rb") as f:
                instance = pickle.load(f)
                bw_files_num += 1
                bw_nodes_num_list.append(nodes_num(instance))
        except Exception as e:
            print(f"Failed to load {file}: {e}")

# Calculate node numbers for sliding puzzle files
sp_nodes_num_list = []
sp_files_num = 0
for dir in dirs_start_with_sp:
    for file in dir.iterdir():
        try:
            with open(file, "rb") as f:
                instance = pickle.load(f)
                sp_files_num += 1
                sp_nodes_num_list.append(nodes_num(instance))
        except Exception as e:
            print(f"Failed to load {file}: {e}")

# Print summary statistics
print(f"Number of block world instances: {bw_files_num}")
print(f"Number of sliding puzzle instances: {sp_files_num}")
print(f"Average number of nodes in block world instances: {sum(bw_nodes_num_list) / len(bw_nodes_num_list)}")
print(f"Average number of nodes in sliding puzzle instances: {sum(sp_nodes_num_list) / len(sp_nodes_num_list)}")
