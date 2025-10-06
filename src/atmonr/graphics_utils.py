"""Various utilities related to computer graphics."""

import torch


def render(
    z_vals: torch.Tensor,
    color: torch.Tensor,
    sigma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Render sampled colors and densities along viewing rays using the Beer-Lambert
    Law. This approximates the true volume integral by assuming its homogeneity between
    subsequent samples.

    Args:
        z_vals: Distances in km of samples along the viewing rays (B, N_samples).
        color: Colors at sampled locations (B, N_samples, N_lambda).
        sigma: Densities per band at sampled locations (B, N_samples, 1 or N_lambda).

    Returns:
        color_map: Collapsed color map as seen from viewing origin (B, N_lambda).
        alpha: Attenuation at each location along rays (B, N_samples, N_lambda).
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
    return color_map, alpha, weights


def render_with_surface(
    z_vals: torch.Tensor,
    color: torch.Tensor,
    sigma: torch.Tensor,
    color_surf: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Render using a surface with assumed infinite density.

    Args:
        z_vals: Distances in km of samples along the viewing rays (B, N_samples).
        color: Colors at sampled locations (B, N_samples, N_lambda).
        sigma: Densities per band at sampled locations (B, N_samples, 1 or N_lambda).
        color_surf: Colors at sampled surface locations (B, N_lambda).

    Returns:
        color_map: Collapsed color map as seen from viewing origin (B, N_lambda).
        alpha: Attenuation at each location along rays (B, N_samples, N_lambda).
        weights: Samples of PDF describing relative volume density for coarse-to-fine
            (B, N_c, 1).
        color_map_atmo: Collapsed color map of only atmospheric samples (B, N_lambda).
        color_map_surf: Color map of only surface samples (B, N_lambda).
    """
    color_map_atmo, alpha, weights = render(z_vals, color, sigma)
    color_map_surf = (1 - alpha).prod(dim=1) * color_surf
    color_map = color_map_atmo + color_map_surf
    return color_map, alpha, weights, color_map_atmo, color_map_surf


def voxel_traversal(
    u: torch.Tensor,
    end: torch.Tensor,
    unique_only: bool = True,
) -> torch.Tensor:
    """Get all the voxels between a starting and ending location. Start and end
    locations are floating point locations on a voxel grid with voxel size 1.
    Implemented for N-dimensional points, though only tested in 2D and 3D.

    Amanatides, John & Woo, Andrew. (1987). A Fast Voxel Traversal Algorithm
    for Ray Tracing. Proceedings of EuroGraphics. 87.

    Args:
        u: Origin point tensor of shape (N, D).
        end: Ray destination tensor of shape (N, D).
        unique_only: If true, return only unique voxels.

    Returns:
        vox_registry: Array of all voxel indices traversed.
    """
    assert u.shape == end.shape and len(u.shape) == 2

    dists = torch.linalg.norm(end - u, dim=-1)[:, None]
    v = (end - u) / dists

    # current & ending voxel index
    vox_idx = torch.floor(u).short()
    vox_idx_end = torch.floor(end).short()
    # sign of the directions needed for various other computations
    sign_v = torch.sign(v).short()
    sign_v_u = sign_v * u

    # value of t at which the ray first leaves this voxel for each axis
    tmax = torch.abs((torch.ceil(sign_v_u) - sign_v_u) / v)
    tmax[tmax.isnan()] = torch.inf
    tmax[vox_idx == vox_idx_end] = torch.inf  # don't move in axes that are already good

    # distance along ray in units of t to equal voxel size
    tdelta = torch.abs(1 / v)

    # registry of visited voxels
    vox_registry = torch.unique(vox_idx, dim=0, sorted=False)
    diff = (vox_idx - vox_idx_end) * sign_v
    # alg termination for each ray
    done = (diff == 0).all(dim=-1) + (diff > 0).any(dim=-1)

    i = 0
    while not done.all():
        # find axis orthogonal to next voxel face crossed
        next_axis = torch.argmin(tmax[~done], dim=-1)
        # increment t up to next crossing
        tmax[~done, next_axis] += tdelta[~done, next_axis]
        # increment voxel idx
        vox_idx[~done, next_axis] += sign_v[~done, next_axis]
        vox_registry = torch.cat([vox_registry, vox_idx[~done]], dim=0)
        diff = (vox_idx[~done] - vox_idx_end[~done]) * sign_v[~done]
        diff_nonneg = diff >= 0
        overshot = (diff > 0).any(dim=-1)
        forbid_axes = diff_nonneg.half()
        forbid_axes[forbid_axes > 0] = torch.inf
        tmax[~done] += forbid_axes
        done[~done] += diff_nonneg.all(dim=-1) + overshot
        i += 1

    if unique_only:
        vox_registry = torch.unique(vox_registry, dim=0, sorted=False)

    return vox_registry
