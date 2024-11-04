import torch
from tqdm import tqdm


def train_with_warmup(model, train_loader, eval_loader, optimizer, epochs, warmup_epochs, max_grad_norm, patience,
                      eval_every, best_model_path, device, logger):
    """
    Train model with warmup using separate train/test loaders and early stopping.
    """
    model = model.to(device)
    criterion = torch.nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=warmup_epochs/epochs
    )

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        nodes_num = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            batch = batch.to(device)
            optimizer.zero_grad()

            predictions = model(batch)
            loss = criterion(predictions, batch.y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item() * len(batch.y) # get sse (sum of squared errors) for the batch
            nodes_num += len(batch.y)

        avg_train_loss = epoch_loss / nodes_num # calculate mse as the sse (sum over all samples) divided by the number of samples

        # Validation phase
        if epoch % eval_every == 0:
            val_loss = evaluate(model, eval_loader, device)

            logger.info(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, '
                       f'Val Loss = {val_loss:.4f}, '
                       f'LR = {scheduler.get_last_lr()[0]:.6f}')

            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f'Early stopping triggered after {epoch+1} epochs')
                    break
        else:
            logger.info(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, '
                       f'LR = {scheduler.get_last_lr()[0]:.6f}')



def evaluate(model, eval_loader, device):
    """ Evaluate the model on a given dataset """
    model.eval()
    total_loss = 0
    num_samples = 0
    criterion = torch.nn.MSELoss()

    with torch.no_grad():
        for batch in eval_loader:
            batch = batch.to(device)
            predictions = model(batch)
            loss = criterion(predictions, batch.y)
            total_loss += loss.item() * len(batch.y) # get sse (sum of squared errors) for the batch
            num_samples += len(batch.y)

    # calculate mse as the sse (sum over all samples) divided by the number of samples
    avg_loss = total_loss / num_samples
    return avg_loss