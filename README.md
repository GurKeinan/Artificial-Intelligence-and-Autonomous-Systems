# Learning to Estimate Search Progress Using Graph Neural Networks

Read our full [report](https://github.com/GurKeinan/Artificial-Intelligence-and-Autonomous-Systems/blob/main/report/submission.pdf) here.

## Table of Contents
- [Learning to Estimate Search Progress Using Graph Neural Networks](#learning-to-estimate-search-progress-using-graph-neural-networks)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Installation](#installation)
  - [Directory Structure](#directory-structure)
  - [Usage](#usage)
  - [Acknowledgements](#acknowledgements)


## Introduction

This project explores a novel approach to estimating search progress in heuristic search algorithms by leveraging Graph Neural Networks (GNNs). Traditional methods often overlook the structural information embedded in the search space, leading to less accurate predictions. Our approach represents the search space as a graph, where nodes correspond to expanded states and edges to state transitions, allowing the GNN to learn patterns in the search trajectory that indicate progress toward the goal.

We evaluate our method on two domains: Blocksworld and Sliding Window Puzzle, using multiple heuristic functions for each.

Our results demonstrate that the GNN-based model can effectively capture search patterns and provide more accurate progress estimates compared to previous methods, even when generalizing to unseen problem instances and across domains. This suggests that incorporating structural information through GNNs leads to more robust and generalizable search progress estimation, potentially improving the efficiency of heuristic search algorithms in practical applications.


## Installation

First, clone the repository to your local machine:

```bash
git clone REPO_URL
```

Then, install the required packages from the requirements file:

```bash
conda env create -n env_name -f requirements.yml
```
This will create a conda environment called "env_name" (replace "env_name" with the desired environment name) and install the specified packages from the requirements file.

## Directory Structure

The directory contains three main directories:

- abstract submission - legacy submission.
- report - contains the project report in latex and pdf format.
- code - contains the code for the project.

Below is a detailed explanation of the code structure:

```bash

├── code
│    ├── benchmarks.py
│    ├── dataset_creation
│    │    ├── general_state.py
│    │    ├── general_A_star.py
│    │    ├── parallel_general_A_star.py
│    │    ├── block_world_generator.py
│    │    ├── block_world_heuristics.py
│    │    ├── sliding_puzzle_generator.py
│    │    ├── sliding_puzzle_heuristics.py
|    |    ├── explore_dataset.py
│    ├── comparing_heuristics
│    │    ├── parallel_comparing_heuristics.py
|    ├── dataset
|    ├── models
|    ├── processed
|    ├── utils.py
|    ├── prepare_graph_dataset.py
|    ├── gnn_training_evaluating.py
|    ├── gnn_architectures.py
|    ├── gnn_both_domains.py
|    └── gnn_out_of_domain.py

```

- benchmarks.py: runs benchmarks on filtered search trees.
- dataset_creation/: contains the dataset creation scripts - problem generators, heuristic functions and A* implementations.
- comparing_heuristics/: contains the comparing heuristics implementation.
- dataset/: contains the generated dataset.
- models/: contains the trained GNN models.
- processed/: contains existing dataloaders to avoid recreating them. Notice this directory will initially be empty.
- utils.py: contains utility functions.
- prepare_graph_dataset.py: contains the functions to prepare the graph dataset.
- gnn_training_evaluating.py: contains the training and evaluation functions for the GNN models.
- gnn_architectures.py: contains the GNN architecture classes.
- gnn_both_domains.py: contains the GNN implementation for both domains training and evaluation.
- gnn_out_of_domain.py: contains the GNN implementation for out of domain training and evaluation.


## Usage

First of all, enter the directory where the code is located:

```bash
cd code
```
We used general Path references throughout the code so there are no specific paths to be specified.

First, one need to create the dataset. For that, we have two options - serial and parallel. For serial, run the following code:

```bash
python dataset_creation/general_A_star.py
```

For parallel, run the following code:

```bash
python dataset_creation/parallel_general_A_star.py
```

Use caution - this will create a lot of files. Make sure you have enough space on your machine or modify the script constants (speaking from painful experience).

After that one can start exploring the code. For benchmarks, run the following code:

```bash
python benchmarks.py
```

For heuristic comparison, run the following code:

```bash
python comparing_heuristics/parallel_comparing_heuristics.py
```

Use caution - the script is parallelized and can use up to 75% of the available CPU cores.

For training and evaluation of the GNN models on both domains, pick a model ("LightGNN" or "HeavyGNN"), change line 17 in gnn_both_domains.py to "LightGNN" or "HeavyGNN" respectively, and run the following code:

```bash
python gnn_both_domains.py
```

For out of domain training and evaluation, run the following code:

```bash
python gnn_out_of_domain.py
```

This code uses the "HeavyGNN" model as default, with slight modifications it can use the "LightGNN" model.


## Acknowledgements

This project was developed by Gur Keinan and Naomi Derel as part of the Artificial Intelligence and Autonomous Systems course at the Technion.

Our work was based on the previous work of [Sudry and Karpas (2022)](https://ojs.aaai.org/index.php/ICAPS/article/view/19821), which we recommend reading for a more in-depth understanding of the problem and the proposed solution.
