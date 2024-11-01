# Learning to Estimate Search Progress Using Graph Neural Networks

## Project Overview

## Table of Contents
- [Introduction](#introduction)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)


## Introduction

This project explores a novel approach to estimating search progress in heuristic search algorithms by leveraging Graph Neural Networks (GNNs). Traditional methods often overlook the structural information embedded in the search space, leading to less accurate predictions. Our approach represents the search space as a graph, where nodes correspond to expanded states and edges to state transitions, allowing the GNN to learn patterns in the search trajectory that indicate progress toward the goal.

We evaluate our method on two domains: STRIPS Blocksworld and Sliding Window Puzzle, using multiple heuristic functions for each.

VARIFY THE FOLLOWING:

Our results demonstrate that the GNN-based model can effectively capture search patterns and provide more accurate progress estimates compared to previous methods, even when generalizing to unseen problem instances and across domains. This suggests that incorporating structural information through GNNs leads to more robust and generalizable search progress estimation, potentially improving the efficiency of heuristic search algorithms in practical applications.

## Directory Structure


## Installation

First, clone the repository to your local machine:

```bash
git clone REPO_URL
```

Next, navigate to the code directory:

```bash
cd code
```

Then, install the required packages:

```bash
pip install -r requirements.yml
```


## Usage

