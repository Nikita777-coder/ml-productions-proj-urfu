from __future__ import annotations

import gzip
import logging
import shutil
import struct
import subprocess
import urllib.request
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from fashion_mnist.config import TrainConfig

logger = logging.getLogger(__name__)

NORMALIZE_MEAN = (0.5,)
NORMALIZE_STD = (0.5,)

FASHION_MNIST_MIRRORS = (
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/",
    "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/",
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
)

# filename, gzip magic/min size, extracted idx magic, extracted min size
FASHION_MNIST_FILES = (
    ("train-images-idx3-ubyte.gz", 1_000_000, 0x00000803, 28 * 28 * 60_000),
    ("train-labels-idx1-ubyte.gz", 10_000, 0x00000801, 60_000),
    ("t10k-images-idx3-ubyte.gz", 100_000, 0x00000803, 28 * 28 * 10_000),
    ("t10k-labels-idx1-ubyte.gz", 1_000, 0x00000801, 10_000),
)


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]
    )


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    try:
        subprocess.run(
            [
                "curl",
                "-L",
                "-f",
                "--retry",
                "3",
                "--retry-all-errors",
                "-A",
                "fashion-mnist-production/1.0",
                "-o",
                str(tmp_path),
                url,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "fashion-mnist-production/1.0"},
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            expected = response.headers.get("Content-Length")
            with tmp_path.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        if expected and tmp_path.stat().st_size != int(expected):
            actual = tmp_path.stat().st_size
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"truncated download {actual} != {expected}") from None
    if not tmp_path.is_file() or tmp_path.stat().st_size == 0:
        raise RuntimeError(f"empty download from {url}")
    tmp_path.replace(destination)


def _extract_gz(archive: Path) -> Path:
    destination = archive.with_suffix("")
    with gzip.open(archive, "rb") as source, destination.open("wb") as handle:
        shutil.copyfileobj(source, handle)
    return destination


def _is_valid_archive(path: Path, min_size: int) -> bool:
    if not path.is_file() or path.stat().st_size < min_size:
        return False
    try:
        with gzip.open(path, "rb") as handle:
            return handle.read(2) == b"\x00\x00"
    except OSError:
        return False


def _is_valid_idx(path: Path, magic: int, min_payload: int) -> bool:
    if not path.is_file() or path.stat().st_size < min_payload:
        return False
    with path.open("rb") as handle:
        header = handle.read(4)
    if len(header) < 4:
        return False
    return struct.unpack(">I", header)[0] == magic


def ensure_fashion_mnist(data_dir: Path) -> Path:
    """Download FashionMNIST archives from a working mirror if needed."""
    datasets.FashionMNIST.mirrors = list(FASHION_MNIST_MIRRORS)
    raw_dir = data_dir / "FashionMNIST" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for filename, min_gz_size, magic, min_payload in FASHION_MNIST_FILES:
        archive = raw_dir / filename
        extracted = archive.with_suffix("")
        if _is_valid_idx(extracted, magic, min_payload):
            continue

        last_error: Exception | None = None
        for mirror in FASHION_MNIST_MIRRORS:
            url = f"{mirror}{filename}"
            try:
                logger.info("Downloading %s", url)
                _download_file(url, archive)
                if not _is_valid_archive(archive, min_gz_size):
                    raise RuntimeError(f"invalid gzip archive from {url}")
                extracted_path = _extract_gz(archive)
                if not _is_valid_idx(extracted_path, magic, min_payload):
                    raise RuntimeError(f"invalid IDX file extracted from {url}")
                last_error = None
                break
            except Exception as error:  # noqa: BLE001
                last_error = error
                archive.unlink(missing_ok=True)
                extracted.unlink(missing_ok=True)
                logger.warning("Failed to download %s: %s", url, error)

        if last_error is not None:
            raise RuntimeError(
                f"Could not download {filename}. Delete {raw_dir} and retry."
            ) from last_error

    return data_dir


def create_dataloaders(config: TrainConfig) -> tuple[DataLoader, DataLoader]:
    """Download FashionMNIST if needed and return train/test loaders."""
    config.data_dir.mkdir(parents=True, exist_ok=True)
    ensure_fashion_mnist(config.data_dir)
    transform = build_transform()

    train_set = datasets.FashionMNIST(
        root=str(config.data_dir),
        train=True,
        download=False,
        transform=transform,
    )
    test_set = datasets.FashionMNIST(
        root=str(config.data_dir),
        train=False,
        download=False,
        transform=transform,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return train_loader, test_loader
