from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fashion_mnist.config import TrainConfig
from fashion_mnist.data import create_dataloaders
from fashion_mnist.device import resolve_device
from fashion_mnist.model import FashionCNN

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate(
    model: FashionCNN,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        predicted = logits.argmax(dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return 100.0 * correct / total if total else 0.0


def load_checkpoint(path: Path, config: TrainConfig, device: torch.device) -> FashionCNN:
    model = FashionCNN(num_classes=config.num_classes, dropout=config.dropout)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved FashionCNN checkpoint")
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    config = TrainConfig()
    device = resolve_device()
    checkpoint = args.checkpoint or config.checkpoint_path

    _, test_loader = create_dataloaders(config)
    model = load_checkpoint(checkpoint, config, device)
    accuracy = evaluate(model, test_loader, device)
    logger.info("Test accuracy: %.2f%% from %s", accuracy, checkpoint)


if __name__ == "__main__":
    main()
