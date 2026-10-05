from pathlib import Path

from fashion_mnist.config import TrainConfig


def test_checkpoint_path() -> None:
    config = TrainConfig(artifact_dir=Path("artifacts"))
    assert config.checkpoint_path == Path("artifacts") / "fashion_cnn.pt"
