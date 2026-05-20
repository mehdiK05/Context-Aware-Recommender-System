import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from . import config
from .datasets import RecommendationDataset
from .models import build_model
from .utils import ensure_dir, set_seed


def _move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        batch = _move_batch(batch, device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, batch["label"])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch["label"])
    return total_loss / len(dataloader.dataset)


def evaluate_loss(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            batch = _move_batch(batch, device)
            logits = model(batch)
            loss = criterion(logits, batch["label"])
            total_loss += loss.item() * len(batch["label"])
    return total_loss / len(dataloader.dataset)


def train_model(
    model_variant,
    train_df,
    val_df,
    metadata,
    save_path,
    epochs=config.EPOCHS,
    batch_size=config.BATCH_SIZE,
    learning_rate=config.LEARNING_RATE,
    patience=config.EARLY_STOPPING_PATIENCE,
):
    set_seed()
    ensure_dir(config.TRAINING_LOGS_DIR)
    ensure_dir(config.MODELS_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_variant, metadata).to(device)
    train_loader = DataLoader(RecommendationDataset(train_df, model_variant), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(RecommendationDataset(val_df, model_variant), batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    best_val = float("inf")
    best_state = None
    stale_epochs = 0
    history = []
    for epoch in tqdm(range(1, epochs + 1), desc=f"Training {model_variant}"):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"{model_variant} epoch {epoch:02d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping {model_variant} after {epoch} epochs.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), save_path)
    history_df = pd.DataFrame(history)
    history_df.to_csv(config.TRAINING_LOGS_DIR / f"{model_variant}_history.csv", index=False)
    return model, history_df

