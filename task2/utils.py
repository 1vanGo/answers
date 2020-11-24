"""Utils for preprocessing and training."""
import numpy as np
import torch


def sigmoid(x):
    """Compute sigmoid."""
    return 1 / (1 + np.exp(-x))


def get_device():
    """Get cuda device if available."""
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def mono_to_color(x):
    """Stack x as x, x, x"""
    return np.stack([x, x, x], axis=-1)


def mixup(data, targets, alpha):
    """Perform mixup augmentation."""
    indices = torch.randperm(data.size(0))
    data2 = data[indices]
    targets2 = targets[indices]

    lam = torch.FloatTensor([np.random.beta(alpha, alpha)])
    data = data * lam + data2 * (1 - lam)
    targets = targets * lam + targets2 * (1 - lam)
    return data, targets
