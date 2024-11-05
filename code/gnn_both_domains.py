"""
Train a GNN model on the full dataset with both domains. Two GNN models are available: HeavyGNN and LightGNN.
"""

from pathlib import Path
from time import time
import sys
import torch
from torch.optim.adamw import AdamW
from gnn_training_evaluating import train_with_warmup, evaluate
from utils import setup_logger, get_pruned_dataloaders
from gnn_architectures import HeavyGNN, LightGNN
from prepare_graph_dataset import get_filtered_dataloaders

# filtering constants
MAX_NODES = 15000
#model constants
MODEL = "HeavyGNN"
HIDDEN_DIM = 256
NUM_LAYERS = 4
HEADS = 4
DROPOUT = 0.2
LAYER_NORM = True
RESIDUAL_FREQUENCY = 2
# optimizer constants
LR = 0.001
WEIGHT_DECAY = 0.01
# training constants
EPOCHS = 50
WARMUP_EPOCHS = 10
BATCH_SIZE = 16
PATIENCE = 5
MAX_GRAD_NORM = 1.0
EVAL_EVERY = 1
TRAIN_RATIO = 0.7
EVAL_RATIO = 0.15
TEST_RATIO = 0.15

repo_root = Path(__file__).resolve().parent
dataset_creation_path = repo_root / "dataset_creation"
sys.path.append(str(dataset_creation_path))

# create models directory if it doesn't exist
models_dir = repo_root / "models"
models_dir.mkdir(exist_ok=True)

# Set up logging
logfile_path = repo_root / "logs" / \
    f"gnn_both_domains_{MODEL}_ci.log"
logger = setup_logger(logfile_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def results_on_loaders(model, train_loader, eval_loader, test_loader):
    total_time_start = time()

    # Evaluate on training set
    start_time = time()
    train_loss = evaluate(model, train_loader, device)
    logger.info("Error on the training set: %f", train_loss)
    logger.info("Runtime for training set evaluation: %f", time() - start_time)

    # Evaluate on validation set
    start_time = time()
    eval_loss = evaluate(model, eval_loader, device)
    logger.info("Error on the validation set: %f", eval_loss)
    logger.info("Runtime for validation set evaluation: %f",
                time() - start_time)

    # Evaluate on test set
    start_time = time()
    test_loss = evaluate(model, test_loader, device)
    logger.info("Error on the test set: %f", test_loss)
    logger.info("Runtime for test set evaluation: %f", time() - start_time)

    # Evaluate on complete dataset
    full_error = (train_loss * len(train_loader.dataset) + eval_loss * len(eval_loader.dataset) + test_loss *
                  len(test_loader.dataset)) / (len(train_loader.dataset) + len(eval_loader.dataset) + len(test_loader.dataset))
    logger.info("Error on the complete dataset: %f", full_error)
    logger.info("Runtime for complete dataset evaluation: %f",
                time() - total_time_start)


def get_eval_ci(model, full_loader, iterations=5):
    """
    Get confidence intervals for the loss and runtime of the model on the full dataset.
    """

    results = []
    durations = []

    for i in range(iterations):
        model.eval()
        new_loader, _, _ = get_pruned_dataloaders(
            full_loader,
            train_ratio=1,
            eval_ratio=0,
            test_ratio=0,
            logger=logger
        )

        start_time = time()
        loss = evaluate(model, new_loader, device)
        duration = time() - start_time

        results.append(loss)
        durations.append(duration)

        logger.info("Iteration %d: Loss: %f, Duration: %f", i, loss, duration)

    # Save results to file
    for value in [results, durations]:
        logger.info("Results: %s", value)

        mean = sum(value) / len(value)
        variance = sum((x - mean) ** 2 for x in value) / len(value)
        ci = 1.96 * (variance / len(value)) ** 0.5
        ci_bounds = (mean - ci, mean + ci)

        logger.info("Mean: %f", mean)
        logger.info("Variance: %f", variance)
        logger.info("Confidence interval: %f", ci)
        logger.info("Confidence interval bounds: %s", ci_bounds)


def main():
    """
    Main function to set up data, initialize the model, and start training.
    """

    # Set random seed for reproducibility
    torch.manual_seed(42)
    logger.info("Using device: %s", device)

    # Load data with smaller batch size
    base_dir = Path(__file__).resolve().parent
    if base_dir.name != "code":
        base_dir = base_dir / "code"
    data_dir = base_dir / "dataset"
    processed_path = base_dir / "processed" \
        / f"Dataloader_{MAX_NODES}.pt"

    # Use smaller batch size
    full_loader = get_filtered_dataloaders(
        root_dir=data_dir,
        processed_path=processed_path,
        batch_size=BATCH_SIZE,
        test_ratio=TEST_RATIO,
        max_nodes=MAX_NODES,    # Added filtering parameters
    )

    logger.info("Full dataset loaded with %d batches", len(full_loader))
    feature_dim = full_loader.dataset.num_features

    # Prune and split dataset
    train_loader, eval_loader, test_loader = get_pruned_dataloaders(
        full_loader,
        train_ratio=TRAIN_RATIO,
        eval_ratio=EVAL_RATIO,
        test_ratio=TEST_RATIO,
        logger=logger
    )
    if MODEL == "HeavyGNN":
        model = HeavyGNN(
            input_dim=feature_dim,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            heads=HEADS,
            dropout=DROPOUT,
            layer_norm=LAYER_NORM,
            residual_frequency=RESIDUAL_FREQUENCY
        )
    else:
        model = LightGNN(
            input_dim=feature_dim,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            layer_norm=LAYER_NORM,
            residual_frequency=RESIDUAL_FREQUENCY
        )

    # Initialize optimizer with weight decay
    optimizer = AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    # Train with warmup and gradient accumulation
    # logger.info("Starting training...")
    # train_with_warmup(
    #     model,
    #     train_loader,
    #     eval_loader,
    #     optimizer,
    #     epochs=EPOCHS,
    #     warmup_epochs=WARMUP_EPOCHS,
    #     max_grad_norm=MAX_GRAD_NORM,
    #     patience=PATIENCE,
    #     eval_every=EVAL_EVERY,
    #     best_model_path=models_dir / f"gnn_both_domains_{MODEL}_best_model.pth",
    #     device=device,
    #     logger=logger
    # )

    # Load the best trained model
    if device == torch.device('cpu'):
        model.load_state_dict(torch.load(models_dir / f"gnn_both_domains_{MODEL}_best_model.pth", map_location=device))
    else:
        model.load_state_dict(torch.load(models_dir / f"gnn_both_domains_{MODEL}_best_model.pth"))
    model.to(device)
    
    logger.info("Model loaded successfully")

    get_eval_ci(model, full_loader, iterations=10)

if __name__ == "__main__":
    main()
