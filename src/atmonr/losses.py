import torch
import torch.nn.functional as F


def dark_loss(pred: torch.Tensor, gt: torch.Tensor, max_i: float) -> torch.Tensor:
    # nerf-in-the-dark
    loss = (((pred - gt) / (pred.detach() + 1e-3 * max_i)) ** 2).mean()
    return loss


def hdr_loss(pred: torch.Tensor, gt: torch.Tensor, max_i: float) -> torch.Tensor:
    return F.mse_loss(torch.log(gt + 1e-3 * max_i), torch.log(pred + 1e-3 * max_i))
    # return F.l1_loss(torch.log(gt + 1e-3 * max_i), torch.log(pred + 1e-3 * max_i))


def l1_loss(pred: torch.Tensor, gt: torch.Tensor, max_i: float) -> torch.Tensor:
    return F.l1_loss(pred / max_i, gt / max_i)


def l1_plus_hdr_loss(pred: torch.Tensor, gt: torch.Tensor, max_i: float) -> torch.Tensor:
    return l1_loss(pred, gt, max_i) + 0.2 * hdr_loss(pred, gt, max_i)


def mse_loss(pred: torch.Tensor, gt: torch.Tensor, max_i: float) -> torch.Tensor:
    return F.mse_loss(pred / max_i, gt / max_i)
