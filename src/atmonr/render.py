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
        z_vals: Distances of samples along the viewing rays (B, N_samples).
        color: Colors at sampled locations(B, N_samples, N_lambda).
        sigma: Densities per band at sampled locations (B, N_samples, N_lambda).

    Returns:
        color_map: Collapsed color map as seen from viewing origin (B, N_lambda).
        weights: Samples of PDF describing relative volume density for coarse-to-fine
            (B, N_c, 1).
    """
    # non-negative density and color
    color = F.relu(color)
    sigma = F.relu(sigma)
    # distances between samples along each viewing ray
    delta = torch.diff(z_vals, dim=1, prepend=z_vals[:, 0:1] * 0)[..., None]
    # append to the start of each ray and interpolate between subsequent densities
    sigma = torch.cat([torch.zeros_like(sigma[:, :1]), sigma], dim=1)
    sigma = (sigma[:, :-1] + sigma[:, 1:]) / 2
    # Beer-Lambert Law: exponential of density integral = attenuation
    alpha = 1 - torch.exp(-sigma * delta)
    # the following is equivalent to alpha blending
    if len(alpha.shape) == 2:
        ones = torch.ones((alpha.shape[0], 1), device=alpha.device)
        weights = (
            alpha
            * torch.cumprod(torch.cat([ones, 1 - alpha + 1e-10], dim=1), dim=1)[:, :-1]
        )[..., None]
    elif len(alpha.shape) == 3:
        ones = torch.ones((alpha.shape[0], 1, alpha.shape[2]), device=alpha.device)
        weights = (
            alpha
            * torch.cumprod(torch.cat([ones, 1 - alpha + 1e-10], dim=1), dim=1)[:, :-1]
        )
    else:
        raise ValueError(
            f"Expected alpha to have 2 or 3 dimensions, but got {len(alpha.shape)} "
            "dimensions."
        )
    color_map = torch.sum(color * weights, dim=1)

    return color_map, weights
