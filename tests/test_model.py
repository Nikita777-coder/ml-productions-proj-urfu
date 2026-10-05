import torch

from fashion_mnist.model import FashionCNN


def test_forward_shape() -> None:
    model = FashionCNN()
    batch = torch.randn(8, 1, 28, 28)
    logits = model(batch)
    assert logits.shape == (8, 10)


def test_parameter_count_is_positive() -> None:
    model = FashionCNN()
    total = sum(parameter.numel() for parameter in model.parameters())
    assert total > 0
