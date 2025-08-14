import torch
import torch.nn.functional as F


def render(
    z_vals: torch.Tensor,
    color: torch.Tensor,
    sigma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render sampled colors and densities along viewing rays using the Beer-Lambert
    Law. This approximates the true volume integral by assuming its homogeneity between
    subsequent samples.

    Args:
        z_vals: Distances in km of samples along the viewing rays (B, N_samples).
        color: Colors at sampled locations(B, N_samples, N_lambda).
        sigma: Densities per band at sampled locations (B, N_samples, 1 or N_lambda).

    Returns:
        color_map: Collapsed color map as seen from viewing origin (B, N_lambda).
        weights: Samples of PDF describing relative volume density for coarse-to-fine
            (B, N_c, 1).
    """
    assert len(z_vals.shape) == 2 and len(color.shape) == 3 and len(sigma.shape) == 3
    assert z_vals.shape == color.shape[:2] and z_vals.shape == sigma.shape[:2]
    z_vals = z_vals.to(dtype=color.dtype)

    # get midpoints between samples along each viewing ray
    z_vals_mid = (z_vals[..., :-1] + z_vals[..., 1:]) / 2
    # prepend the ray origin and append final z value
    z_vals_mid = torch.cat([z_vals[..., :1] * 0, z_vals_mid, z_vals[..., -1:]], dim=-1)
    # deltas correspond to a voronoi partition of the samples along each ray
    delta = torch.diff(z_vals_mid, dim=-1)[..., None]

    # Beer-Lambert Law: exponential of density integral = attenuation
    alpha = 1 - torch.exp(-sigma * delta)
    # the following is equivalent to alpha blending
    ones = torch.ones(
        (alpha.shape[0], 1, alpha.shape[2]), device=alpha.device, dtype=alpha.dtype
    )
    weights = (
        alpha
        * torch.cumprod(torch.cat([ones, 1 - alpha + 1e-10], dim=1), dim=1)[:, :-1]
    )

    color_map = torch.sum(color * weights, dim=1)
    return color_map, weights
