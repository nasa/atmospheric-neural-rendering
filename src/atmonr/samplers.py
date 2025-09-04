from typing import Mapping

import torch

from atmonr.geospatial.wgs_84 import cartesian_to_horizontal


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
    t_in_bin = (
        torch.rand((ray_batch["origin"].shape[0], n_bins), device=device)
        if random
        else 0.5
    )
    z_vals = (bins[:, :-1] + t_in_bin / n_bins) * ray_batch["len"][:, None]

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
    t_in_bin = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t_in_bin * (bins_g[..., 1] - bins_g[..., 0]).detach()

    z_vals, _ = torch.sort(torch.cat([z_vals_c, samples], -1), -1)

    # compute 3D locations
    pts = ray_batch["origin"][:, None] + ray_batch["dir"][:, None] * z_vals[..., None]

    return pts, z_vals


def sample_biased_bins(
    ray_batch: Mapping[str, torch.Tensor],
    n_bins: int,
    ray_origin_height: float,
    subsurface_depth: float,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample points uniformly from variably-sized bins along a viewing ray. Points near
    the surface are more commonly selected. The bins use a probability distribution
    which increases linearly from the ray origin until the approximate intersection with
    the surface, and stays constant from the surface until the ray terminates. The alpha
    parameter is the ratio between the probability at ray origin and the probability at
    the surface. The unnormalized pdf over the interval [0, surface] is a trapezoid with
    side lengths of 1 and alpha, and is constant over the interval [surface, ray_len].
    The binning occurs in the input space, not in the pdf space, so the bins are of
    unequal extents.

    Note: this function assumes that rays are normalized and the scene is already
    cropped to near/far bounds.

    Args:
        ray_batch: A batch of rays containing the origins, direction vectors, and
            lengths of the rays to sample.
        n_bins: How many bins to stratify each ray into. The default for NeRF coarse
            sampling is 64.
        ray_origin_height: Height above sea level (in meters) at which ray origins were
            constructed.
        subsurface_depth: Height below sea level (in meters) at which rays terminate.
        alpha: Probability at ray_origin_height divided by probability at surface.

    Returns:
        pts: Sampled 3D points, of shape (batch_size, n_bins, 3).
        z_vals: Distances along each ray that points were taken, of shape (batch_size,
            n_bins).
    """
    assert alpha >= 0 and alpha <= 1

    # ratio of above-surface to total length
    r_x = ray_origin_height / (ray_origin_height + subsurface_depth)
    # solution of inverse of cdf for r_x
    r_y = (r_x * (alpha + 1) / 2) / (r_x * (alpha - 1) / 2 + 1)
    # normalization term := sum of cdf over [0, 1]
    norm_term = (r_x * (alpha - 1) / 2) + 1

    device = ray_batch["origin"].device

    # chop the space up into uniformly-sized bins of shape (1, n_bins)
    bins = torch.linspace(0, 1, n_bins + 1, device=device)[None]

    # randomly sample the uniform distribution
    t_in_bin = torch.rand((ray_batch["origin"].shape[0], n_bins), device=device)
    z_vals_flat = bins[:, :-1] + t_in_bin / n_bins

    # use the solution for the inverse of the cdf for binned samples
    mask = z_vals_flat <= r_y
    z_vals = torch.zeros_like(z_vals_flat)
    z_vals[mask] = (
        -alpha
        + torch.sqrt(alpha**2 + 2 * (1 - alpha) * norm_term * z_vals_flat[mask] / r_x)
    ) * (r_x / (1 - alpha))
    z_vals[~mask] = r_x + (1 - r_x) * (z_vals_flat[~mask] - r_y) / (1 - r_y)
    z_vals *= ray_batch["len"][:, None]

    # get the points along those rays
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
