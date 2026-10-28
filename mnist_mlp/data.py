from pathlib import Path

import torch
from torch.utils.data import TensorDataset
from torchvision.datasets import MNIST


def stratified_split(labels, seed=42, validation_fraction=1 / 6):
    """Her sınıftan aynı oranda, ayrık ve yeniden üretilebilir indeksler."""
    generator = torch.Generator().manual_seed(seed)
    train, validation = [], []
    for label in range(10):
        indices = torch.where(labels == label)[0]
        indices = indices[torch.randperm(len(indices), generator=generator)]
        count = round(len(indices) * validation_fraction)
        validation.append(indices[:count])
        train.append(indices[count:])
    train, validation = torch.cat(train), torch.cat(validation)
    return (
        train[torch.randperm(len(train), generator=generator)],
        validation[torch.randperm(len(validation), generator=generator)],
    )


def training_data(root, seed):
    raw = MNIST(root=root, train=True, download=True)
    train_idx, val_idx = stratified_split(raw.targets, seed)
    train_pixels = raw.data[train_idx].float().unsqueeze(1) / 255
    mean, std = train_pixels.mean().item(), train_pixels.std(unbiased=False).item()
    validation_pixels = raw.data[val_idx].float().unsqueeze(1) / 255
    train = TensorDataset((train_pixels - mean) / std, raw.targets[train_idx])
    validation = TensorDataset((validation_pixels - mean) / std, raw.targets[val_idx])
    return train, validation, train_idx, val_idx, {"mean": mean, "std": std}


def test_data(root, normalization):
    raw = MNIST(root=Path(root), train=False, download=True)
    pixels = raw.data.float().unsqueeze(1) / 255
    return TensorDataset(
        (pixels - normalization["mean"]) / normalization["std"], raw.targets
    )
