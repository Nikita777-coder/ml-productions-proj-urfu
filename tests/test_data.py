from pathlib import Path

from fashion_mnist.data import FASHION_MNIST_MIRRORS, _is_valid_archive, build_transform


def test_transform_converts_image_to_tensor() -> None:
    from PIL import Image

    image = Image.new("L", (28, 28), color=128)
    tensor = build_transform()(image)
    assert tuple(tensor.shape) == (1, 28, 28)
    assert tensor.min() >= -1
    assert tensor.max() <= 1


def test_archive_check_rejects_truncated_file(tmp_path: Path) -> None:
    broken = tmp_path / "train-images-idx3-ubyte.gz"
    broken.write_bytes(b"not-a-gzip")
    assert not _is_valid_archive(broken, 1_000_000)


def test_mirrors_prefer_https() -> None:
    assert FASHION_MNIST_MIRRORS[0].startswith("https://")
