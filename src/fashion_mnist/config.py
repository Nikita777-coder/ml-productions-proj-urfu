from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters and filesystem paths for training."""

    data_dir: Path = Path("data")
    artifact_dir: Path = Path("artifacts")
    batch_size: int = 64
    epochs: int = 5
    learning_rate: float = 1e-3
    dropout: float = 0.25
    num_classes: int = 10
    seed: int = 42
    num_workers: int = 0

    @property
    def checkpoint_path(self) -> Path:
        return self.artifact_dir / "fashion_cnn.pt"
