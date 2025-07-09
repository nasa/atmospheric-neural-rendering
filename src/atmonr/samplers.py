from typing import Mapping

import torch

from atmonr.geometry import cartesian_to_horizontal


def sample_uniform_bins(
    ray_batch: Mapping[str, torch.Tensor],
    n_bins: int = 64,
    random: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample points uniformly from evenly-sized bins along a viewing ray.

    Note: this function assumes that rays are normalized and the scene is already
    cropped to near/far bounds.

    Args:
        ray_batch: A batch of rays containing the origins, direction vectors, and
            lengths of the rays to sample.
        n_bins: How many bins to stratify each ray into. The default for NeRF coarse
            sampling is 64.
        random: Whether to sample randomly within each bin (if True) or just take the
            midpoint.

    Returns:
        pts: Sampled 3D points, of shape (batch_size, n_bins, 3).
        z_vals: Distances along each ray that points were taken, of shape (batch_size,
            n_bins).
    """
    device = ray_batch["origin"].device

    # chop the space up into uniformly-sized bins of shape (1, n_bins)
    bins = torch.linspace(0, 1, n_bins + 1, device=device)[None]

    # either randomly sample the uniform distribution or take the middle value
    t = (
        torch.rand((ray_batch["origin"].shape[0], n_bins), device=device)
        if random
        else 0.5
    )
    z_vals = (bins[:, :-1] + t / n_bins) * ray_batch["len"][:, None]

    # get the points along those rays
    pts = ray_batch["origin"][:, None] + ray_batch["dir"][:, None] * z_vals[..., None]

    return pts, z_vals


def sample_pdf(
    ray_batch: Mapping[str, torch.Tensor],
    pdf_discrete: torch.Tensor,
    z_vals_c: torch.Tensor,
    n_samples: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample from a (discrete) probability density function along a ray. Used by the
    fine network.

    Args:
        ray_batch: A batch of rays containing the origins, direction vectors, and
            lengths of the rays to sample.
        pdf_discrete: Values of the pdf at discrete locations along each ray, of shape
            (batch_size, N)
        z_vals_c: Distances along the viewing rays of the coarse samples.
        n_samples: How many samples to take along each ray.

    Returns:
        pts: Sampled 3D points, of shape (batch_size, n_bins, 3)
        z_vals: Distances along each ray that points were taken, of shape (batch_size,
            n_bins)
    """
    pdf_discrete = pdf_discrete[:, 1:-1, 0]
    pdf = (pdf_discrete + 1e-8) / torch.sum(
        pdf_discrete + 1e-8, dim=1, keepdim=True
    )  # normalize
    cdf = torch.cumsum(pdf, dim=1)  # integrate
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=1)

    # Invert CDF
    u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device=cdf.device)
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    z_vals_mid = 0.5 * (z_vals_c[..., 1:] + z_vals_c[..., :-1])

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(z_vals_mid.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-8, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0]).detach()

    z_vals, _ = torch.sort(torch.cat([z_vals_c, samples], -1), -1)

    # compute 3D locations
    pts = ray_batch["origin"][:, None] + ray_batch["dir"][:, None] * z_vals[..., None]

    return pts, z_vals


def append_heights(
    pts: torch.Tensor,
    ray_origin_height: float,
    scale: float,
    offset: torch.Tensor,
) -> torch.Tensor:
    """Append normalized height values to position vectors as an optional additional
    input to a neural rendering model.

    Args:
        pts: Array of 3D points in normalized scene frame, of shape (batch_size,
            num_samples, 3).
        ray_origin_height: Height above sea level (in meters) at which ray origins were
            constructed.

    Returns:
        pts_alt: Original points array with normalized ellipsoidal height appended as a
            redundant 4th component.
    """
    # unnormalize points to get them in WGS-84 Cartesian frame
    xyz = pts.double() * scale + offset[None, None]
    # transform cartesian -> horizontal to get ellipsoidal height
    _, _, alt = cartesian_to_horizontal(xyz[..., 0], xyz[..., 1], xyz[..., 2])
    # normalize by ray_origin_height
    alt = (alt / ray_origin_height).float()
    # concatenate and return
    pts_alt = torch.cat([pts, alt[..., None]], dim=-1)
    return pts_alt
