import torch
import torch.nn.functional as F


def hdr_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return (((pred - gt) / (pred.detach() + 1e-3)) ** 2).mean()
