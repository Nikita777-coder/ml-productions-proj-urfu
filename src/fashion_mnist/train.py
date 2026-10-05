from __future__ import annotations

import argparse
import logging

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from fashion_mnist.config import TrainConfig
from fashion_mnist.data import create_dataloaders
from fashion_mnist.device import resolve_device, seed_everything
from fashion_mnist.evaluate import evaluate
from fashion_mnist.model import FashionCNN

logger = logging.getLogger(__name__)


def train_one_epoch(
    model: FashionCNN,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Adam,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    batches = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        batches += 1
    return running_loss / batches if batches else 0.0


def train(config: TrainConfig) -> float:
    seed_everything(config.seed)
    device = resolve_device()
    config.artifact_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = create_dataloaders(config)
    model = FashionCNN(num_classes=config.num_classes, dropout=config.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    best_accuracy = 0.0
    logger.info("Training on %s for %s epochs", device, config.epochs)
    for epoch in range(1, config.epochs + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        accuracy = evaluate(model, test_loader, device)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), config.checkpoint_path)
        logger.info(
            "Epoch %s/%s loss=%.4f accuracy=%.2f%% best=%.2f%%",
            epoch,
            config.epochs,
            loss,
            accuracy,
            best_accuracy,
        )
    logger.info("Best checkpoint saved to %s", config.checkpoint_path)
    return best_accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FashionCNN on FashionMNIST")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    config = TrainConfig(
        epochs=args.epochs or TrainConfig.epochs,
        batch_size=args.batch_size or TrainConfig.batch_size,
        learning_rate=args.lr or TrainConfig.learning_rate,
    )
    train(config)


if __name__ == "__main__":
    main()
