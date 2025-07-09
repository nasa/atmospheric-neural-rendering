"""A set of geometry utilities for working with viewing rays in various 3D reference
frames. This repository uses the following frames of reference:
1) WGS-84 ellipsoid, horizontal coordinates, EPSG code: 4326 https://epsg.io/4326
2) WGS-84 ellipsoid, Cartesian coordinates, EPSG code: 4978 https://epsg.io/4978
3) Normalized Cartesian scene coordinates, with +X=East, +Y=North, and +Z=up for the
    center pixel, and locations centered around the scene center and max-min normalized
    between -1 and 1.
"""

import numpy as np
import torch

# WGS-84 constants
WGS_84_A = 6378137.0  # semimajor axis
WGS_84_B = 6356752.314245  # semiminor axis
WGS_84_E = (WGS_84_A**2 - WGS_84_B**2) / (WGS_84_A**2)  # first eccentricity
WGS_84_E2 = (WGS_84_A**2 - WGS_84_B**2) / (WGS_84_B**2)  # second eccentricity
WGS_84_F = (WGS_84_A - WGS_84_B) / WGS_84_A  # flattening


def horizontal_to_cartesian(
    lat: torch.Tensor,
    lon: torch.Tensor,
    alt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert from WGS-84 geographical coordinates to Cartesian coordinates.
    This corresponds to epsg:4326 -> epsg:4978. Use caution with precision! Using 32-bit
    precision will lead to errors of up to a few meters. If greater accuracy is needed,
    use 64-bit tensors.

    Args:
        lat: Geographic latitudes.
        lon: Geographic longitudes.
        alt: Ellipsoidal height (surface altitude).

    Returns:
        x: x-coordinate in WGS-84 Cartesian.
        y: y-coordinate in WGS-84 Cartesian.
        z: z-coordinate in WGS-84 Cartesian.
    """
    assert lat.shape == lon.shape and lat.shape == alt.shape
    shp = lat.shape
    lat = lat.flatten() * np.pi / 180
    lon = lon.flatten() * np.pi / 180
    alt = alt.flatten()
    N = WGS_84_A / torch.sqrt(1 - (WGS_84_E * torch.sin(lat) ** 2))
    x = (N + alt) * torch.cos(lat) * torch.cos(lon)
    y = (N + alt) * torch.cos(lat) * torch.sin(lon)
    z = (N * (1 - WGS_84_E) + alt) * torch.sin(lat)
    return x.view(shp), y.view(shp), z.view(shp)


def cartesian_to_horizontal(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert from WGS-84 Cartesian coordinates to geographical coordinates.
    This corresponds to epsg:4978 -> epsg:4326. Use caution with precision! Using 32-bit
    precision will lead to errors of up to 5-6 meters in the ellipsoidal height. If
    greater accuracy is needed, use 64-bit tensors.

    This uses the first-order approximation introduced in the following:
    Bowring, B. R. (1976). TRANSFORMATION FROM SPATIAL TO GEOGRAPHICAL COORDINATES.
        Survey Review, 23(181), 323–327. https://doi.org/10.1179/sre.1976.23.181.323

    Args:
        x: x-coordinate in WGS-84 Cartesian.
        y: y-coordinate in WGS-84 Cartesian.
        z: z-coordinate in WGS-84 Cartesian.

    Returns:
        lat: Geographic latitudes.
        lon: Geographic longitudes.
        alt: Ellipsoidal height (surface altitude).
    """
    assert x.shape == y.shape and x.shape == z.shape
    shp = x.shape
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    lon = torch.atan2(y, x)
    D = torch.sqrt(x**2 + y**2)  # horizontal component
    # parametric lat
    u = torch.atan2(z / D, torch.zeros_like(x) + WGS_84_A / WGS_84_B)
    # first iteration of approximation
    lat = torch.atan2(
        z + (WGS_84_E2 * WGS_84_B) * (torch.sin(u) ** 3),
        D - (WGS_84_E * WGS_84_A) * (torch.cos(u) ** 3),
    )
    # solve for height
    N = WGS_84_A / torch.sqrt(1 - (WGS_84_E * torch.sin(lat) ** 2))
    alt = x / (torch.cos(lat) * torch.cos(lon)) - N
    return lat.view(shp) * 180 / torch.pi, lon.view(shp) * 180 / torch.pi, alt.view(shp)


def horizontal_coords_to_rot_mtx(
    theta: torch.Tensor, phi: torch.Tensor
) -> torch.Tensor:
    """Get a rotation matrix from horizontal coordinates (zenith, azimuth).

    Args:
        theta: Array of zenith angles, in degrees (0 to 180).
        phi: Array of azimuth angles, in degrees (-180 to 180).

    Returns:
        rot_mtx: Rotation matrix.
    """
    assert len(theta.shape) == 1 and theta.shape == phi.shape
    # convert to radians and flip sign of rotation to match 3D rotation convention
    theta, phi = -theta * torch.pi / 180, -phi * torch.pi / 180

    # compute sin/cos
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    sin_phi = torch.sin(phi)
    cos_phi = torch.cos(phi)
    zeros = torch.zeros_like(theta)

    # construct rotation matrix
    rot_mtx = torch.stack(
        [
            torch.stack([cos_phi, -sin_phi * cos_theta, sin_phi * sin_theta], dim=1),
            torch.stack([sin_phi, cos_phi * cos_theta, -cos_phi * sin_theta], dim=1),
            torch.stack([zeros, sin_theta, cos_theta], dim=1),
        ],
        dim=1,
    )
    return rot_mtx


def horizontal_coords_to_dirvecs(
    theta: torch.Tensor, phi: torch.Tensor
) -> torch.Tensor:
    """Transform horizontal coordinates (zenith, azimuth) to direction vectors.

    Args:
        theta: Array of zenith angles, in degrees (0 to 180).
        phi: Array of azimuth angles, in degrees (-180 to 180).

    Returns:
        dirs: Array of direction vectors.
    """
    assert theta.shape == phi.shape
    shp = theta.shape
    theta, phi = theta.flatten(), phi.flatten()

    # get direction vectors in +z = up frame
    dirs = torch.zeros_like(theta)
    dirs = torch.stack([dirs, dirs, dirs + 1], dim=-1)

    # get rotation matrix
    rot_mtx = horizontal_coords_to_rot_mtx(theta, phi)

    # apply rotation matrix
    dirs = rot_mtx @ dirs[..., None]
    return dirs.view(*tuple(list(shp) + [3]))


def dirvecs_to_horizontal_coords(
    dirs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform direction vectors to horizontal coordinates (zenith, azimuth).

    Args:
        dirs: Array of direction vectors.

    Returns:
        theta: Array of zenith angles, in degrees (0 to 180).
        phi: Array of azimuth angles, in degrees (-180 to 180).
    """
    assert len(dirs.shape) > 1 and dirs.shape[-1] == 3
    dirs = dirs.view(-1, 3)

    # compute zenith, azimuth with arctan2
    theta = torch.atan2(torch.linalg.norm(dirs[..., :2]), dirs[..., 2])
    phi = -torch.atan2(dirs[..., 0], -dirs[..., 1])

    # convert to degrees
    theta = (theta * 180 / torch.pi) % 360
    phi = (phi * 180 / torch.pi) % 360 - 180

    return theta, phi


def compose_dirs_and_surface_normals(
    dirs: torch.Tensor,
    lat: torch.Tensor,
    lon: torch.Tensor,
) -> torch.Tensor:
    """Compose directions (in +z = up frame) and latitude / longitude to get directions
    in WGS-84 Cartesian frame.

    Args:
        dirs: Array of direction vectors.
        lat: Array of latitudes.
        lon: Array of longitudes.

    Returns:
        rot_dirs: Array of direction vectors in WGS-84 Cartesian frame.
    """
    assert (
        len(dirs.shape) > 1 and dirs.shape[:-1] == lat.shape and lat.shape == lon.shape
    )
    rot_mtx = horizontal_coords_to_rot_mtx(90 - lat, 90 - lon).to(dtype=dirs.dtype)
    # add a 180 z-rotation because the WGS convention has +X through the prime meridian,
    #   +Y east, and +Z through the north pole, while our scene convention is to have +X
    #   east, Y north, and Z up.
    z_rot_mtx = torch.FloatTensor(
        [
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1],
        ]
    )[None].to(device=dirs.device, dtype=dirs.dtype)
    rot_dirs = (rot_mtx @ (z_rot_mtx @ dirs[..., None]))[..., 0]
    return rot_dirs


def get_rays(
    lat: torch.Tensor,
    lon: torch.Tensor,
    alt: torch.Tensor,
    thetav: torch.Tensor,
    phiv: torch.Tensor,
    ray_origin_height: float,
    subsurface_depth: float,
    tol: float = 10.0,
    max_iters: int = 20,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get an array of rays and an array of ray lengths from satellite data arrays.
    P is the number of pixels in the satellite product, and A is the number of angles.

    Args:
        lat: Latitude array of shape (P, A).
        lon: Longitude array of shape (P, A).
        alt: Surface altitude array of shape (P, A).
        thetav: View zenith array of shape (P, A).
        phiv: View azimuth array of shape (P, A).
        ray_origin_height: Height above surface (in meters) at which to construct
            ray origins.
        subsurface_depth: Depth below surface (in meters) at which to terminate rays.
        tol: Tolerance (in meters) for error in ray origins.
        max_iters: Max number of iterations for finding the ray origins.

    Returns:
        origins: Array of ray origins of shape (P * A, 3).
        comp_dirs: Array of direction vectors in WGS-84 Cartesian frame, of shape
            (P * A, 3).
        lens: Lengths of the ray segments from top of atmosphere to the surface, of
            shape (P * A).
    """

    # convert from WGS-84 lat/lon/alt to Cartesian
    x, y, z = horizontal_to_cartesian(lat.double(), lon.double(), alt.double())
    xyz = torch.stack([x, y, z], dim=-1).float()

    # get direction vectors in right-handed coordinates where +y = north, +z = up
    dirs = horizontal_coords_to_dirvecs(thetav.double(), phiv.double())

    # transform direction vectors from +z: ellipsoid surface normal to +z: North pole
    comp_dirs = compose_dirs_and_surface_normals(
        dirs.view(-1, 3), lat.flatten(), lon.flatten()
    )
    # flip the direction to have top of atmosphere as origin
    comp_dirs = -comp_dirs.view(dirs.shape)

    # # construct origins at approximately ray_origin_height meters from the surface
    # surface_to_origin_lens = ray_origin_height / torch.cos(
    #     thetav * torch.pi / 180
    # ).view(comp_dirs.shape[:2])
    # # add subsurface depth to lengths when sampling rays
    # subsurface_lens = subsurface_depth / torch.cos(thetav * torch.pi / 180).view(
    #     comp_dirs.shape[:2]
    # )

    # iteratively solve for the origin point of rays
    surface_to_origin_lens = (ray_origin_height - alt) / torch.cos(
        thetav * torch.pi / 180
    ).view(comp_dirs.shape[:-1]).double()
    xyz2 = xyz - surface_to_origin_lens[..., None] * comp_dirs
    _, _, alt_check = cartesian_to_horizontal(xyz2[..., 0], xyz2[..., 1], xyz2[..., 2])
    err = torch.abs(ray_origin_height - alt_check)
    iters = 0
    while iters < max_iters and (err > tol).any():
        surface_to_origin_lens = surface_to_origin_lens * ray_origin_height / alt_check
        xyz2 = xyz - surface_to_origin_lens[..., None] * comp_dirs
        _, _, alt_check = cartesian_to_horizontal(
            xyz2[..., 0], xyz2[..., 1], xyz2[..., 2]
        )
        err = torch.abs(ray_origin_height - alt_check)
        iters += 1

    # solve for subsurface (end) point of rays, but only guess once
    subsurface_lens = (subsurface_depth + alt) / torch.cos(
        thetav * torch.pi / 180
    ).view(comp_dirs.shape[:-1])

    # turn back to floats
    surface_to_origin_lens = surface_to_origin_lens
    subsurface_lens = subsurface_lens

    lens = subsurface_lens + surface_to_origin_lens

    origins = (xyz - comp_dirs * surface_to_origin_lens[..., None]).view(-1, 3)
    comp_dirs = comp_dirs.view(-1, 3)
    lens = lens.flatten()

    return origins.float(), comp_dirs.float(), lens.float()


def filter_rays(
    ray_origin: torch.Tensor,
    ray_dir: torch.Tensor,
    ray_rad: torch.Tensor,
) -> torch.Tensor:
    """Filter out rays that have any invalid position, any invalid direction, or no
    valid intensity data.

    Args:
        ray_origin: Ray origins.
        ray_dir: Ray directions.
        ray_rad: Radiances associated with each ray.

    Returns:
        valid: Mask of valid rays.
    """
    pos_nan = ray_origin.isnan().any(dim=1)
    dirs_nan = ray_dir.isnan().any(dim=1)
    color_nan = ray_rad.isnan()
    valid = (~pos_nan) * (~dirs_nan) * (~color_nan)
    return valid


def normalize_rays(
    ray_origin: torch.Tensor,
    ray_dir: torch.Tensor,
    ray_len: torch.Tensor,
) -> tuple[torch.Tensor, float, torch.Tensor]:
    """Normalize into [-1, 1]^3, keeping track of the scale factor and offset.

    Args:
        ray_origin: Ray origins before normalization.
        ray_dir: Ray directions.
        ray_len: Lengths of the ray segments from top of atmosphere to the surface.

    Returns:
        ray_origin_norm: Normalized ray data.
        scale: Scale factor used to normalize the ray data.
        offset: Translational offset used to normalize the ray data.
    """
    xyz = torch.cat([ray_origin, ray_origin + ray_dir * ray_len[:, None]], dim=0)
    xyz_max = xyz.max(dim=0)[0].double()
    xyz_min = xyz.min(dim=0)[0].double()
    scale = ((xyz_max - xyz_min).max() / 2).item()
    offset = (xyz_max + xyz_min) / 2
    ray_origin_norm = torch.clamp((ray_origin - offset) / scale, -1, 1).float()
    return ray_origin_norm, scale, offset


def vincenty_distance(
    latlon1: tuple[torch.Tensor, torch.Tensor] | torch.Tensor,
    latlon2: tuple[torch.Tensor, torch.Tensor] | torch.Tensor,
    tol: float = 1e-12,
    max_iters: int = 10,
) -> tuple[float | torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute geodesic distance using Vincenty's formulae and the WGS-84 ellipsoid.
    This method assumes inputs and outputs are tensors on GPU.

    See: https://en.wikipedia.org/wiki/Vincenty%27s_formulae#Inverse_problem
    Args:
        latlon1: Starting points' latitudes and longitudes, as either a tuple or a
            (2, ...) tensor.
        latlon2: Destination points' latitudes and longitudes, as either a tuple or a
            (2, ...) tensor.
        tol: Tolerance in meters. When the updates are less than tol, the iteration
            ends (default: 1e-12).
        max_iters: Maximum number of iterations to perform.

    Returns:
        s: Geodesic distance in meters between provided points.
        alpha1: Forward azimuths at starting points.
        alpha2: Forward azimuths at destination points.
    """
    assert isinstance(latlon1, tuple) or isinstance(latlon1, torch.Tensor)
    assert isinstance(latlon2, tuple) or isinstance(latlon2, torch.Tensor)
    assert isinstance(tol, float)
    assert isinstance(max_iters, int)

    lat1, lat2 = latlon1[0] * torch.pi / 180, latlon2[0] * torch.pi / 180
    lon1, lon2 = latlon1[1] * torch.pi / 180, latlon2[1] * torch.pi / 180
    U1 = torch.atan((1 - WGS_84_F) * torch.tan(lat1))
    U2 = torch.atan((1 - WGS_84_F) * torch.tan(lat2))
    L = lon2 - lon1

    lambd = L
    lambd_diff = torch.FloatTensor([1000]).to(lat1.device)
    num_iters = 0

    sin_sigma, cos_sigma, sigma, cos2_alpha, cos_2sigmam = 0, 0, 0, 0, 0

    while (torch.abs(lambd_diff) > tol).any():
        if num_iters > max_iters:
            raise Warning(
                f"Exceeded {max_iters} iterations without lambda changing by less than "
                f"{tol:.1e}"
            )

        sin_sigma = torch.sqrt(
            (torch.cos(U2) * torch.sin(lambd)) ** 2
            + (
                torch.cos(U1) * torch.sin(U2)
                - torch.sin(U1) * torch.cos(U2) * torch.cos(lambd)
            )
            ** 2
        )
        cos_sigma = torch.sin(U1) * torch.sin(U2) + torch.cos(U1) * torch.cos(
            U2
        ) * torch.cos(lambd)
        sigma = torch.atan2(sin_sigma, cos_sigma)

        sin_alpha = torch.cos(U1) * torch.cos(U2) * torch.sin(lambd) / sin_sigma
        cos2_alpha = 1 - sin_alpha**2
        cos_2sigmam = cos_sigma - (2 * torch.sin(U1) * torch.sin(U2)) / cos2_alpha

        C = (WGS_84_F / 16) * cos2_alpha * (4 + WGS_84_F * (4 - 3 * cos2_alpha))
        lambd_i = L + (1 - C) * WGS_84_F * sin_alpha * (
            sigma
            + C * sin_sigma * (cos_2sigmam + C * cos_sigma * (-1 + 2 * cos_2sigmam**2))
        )
        lambd_diff = lambd_i - lambd
        lambd = lambd_i
        num_iters += 1

    u2 = cos2_alpha * (WGS_84_A**2 - WGS_84_B**2) / WGS_84_B**2
    A = 1 + (u2 / 16384) * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    B = (u2 / 1024) * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
    delta_sigma = (
        B
        * sin_sigma
        * (
            cos_2sigmam
            + (1 / 4)
            * B
            * (
                cos_sigma * (-1 + 2 * cos_2sigmam**2)
                - (1 / 6)
                * B
                * cos_2sigmam
                * (-3 + 4 * sin_sigma**2)
                * (-3 + 4 * cos_2sigmam**2)
            )
        )
    )

    s = WGS_84_B * A * (sigma - delta_sigma)
    alpha1 = torch.atan2(
        torch.cos(U2) * torch.sin(lambd),
        torch.cos(U1) * torch.sin(U2)
        - torch.sin(U1) * torch.cos(U2) * torch.cos(lambd),
    )
    alpha2 = torch.atan2(
        torch.cos(U1) * torch.sin(lambd),
        -torch.sin(U1) * torch.cos(U2)
        + torch.cos(U1) * torch.sin(U2) * torch.cos(lambd),
    )

    return s, alpha1 * 180 / np.pi, alpha2 * 180 / np.pi


def vincenty_point_along_geodesic(
    latlon1: tuple[torch.Tensor, torch.Tensor] | torch.Tensor,
    alpha1: torch.Tensor,
    s: float | torch.Tensor,
    tol: float = 1e-6,
    max_iters: int = 10,
) -> tuple[tuple[torch.Tensor, torch.Tensor] | torch.Tensor, torch.Tensor]:
    """Compute destination locations along geodesics defined by starting locations,
    azimuth angles, and distances.

    See: https://en.wikipedia.org/wiki/Vincenty%27s_formulae#Direct_problem

    Args:
        latlon1: Starting points' latitudes and longitudes, as either a tuple or a
            (2, ...) tensor.
        alpha1: Forward azimuths at initial points.
        s: Distances to travel along the geodesics.
        tol: Tolerance in meters. When the updates are less than tol, the iteration
            ends (default: 1e-12).
        max_iters: Maximum number of iterations to perform.

    Returns:
        s: Geodesic distance in meters between provided points.
        alpha1: Forward azimuths at starting points.
        alpha2: Forward azimuths at destination points.
    """
    assert isinstance(latlon1, tuple) or isinstance(latlon1, torch.Tensor)
    assert isinstance(alpha1, torch.Tensor)
    assert isinstance(s, torch.Tensor)
    assert isinstance(tol, float)
    assert isinstance(max_iters, int)

    if isinstance(alpha1, float):
        alpha1 = torch.FloatTensor([alpha1]).to(latlon1[0].device)
    if isinstance(s, float):
        s = torch.FloatTensor([s]).to(latlon1[0].device)

    lat1, lon1 = latlon1[0] * torch.pi / 180, latlon1[1] * torch.pi / 180
    alpha1 = alpha1 * torch.pi / 180

    U1 = torch.atan((1 - WGS_84_F) * torch.tan(lat1))
    sigma1 = torch.atan2(torch.tan(U1), torch.cos(alpha1))
    sin_alpha = torch.cos(U1) * torch.sin(alpha1)
    u2 = (1 - sin_alpha**2) * (WGS_84_A**2 - WGS_84_B**2) / WGS_84_B**2
    A = 1 + (u2 / 16384) * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    B = (u2 / 1024) * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))

    sigma = s / (WGS_84_B * A)
    sigma_diff = torch.FloatTensor([1000]).to(lat1.device)
    num_iters = 0

    cos_2sigmam = 0

    while (torch.abs(sigma_diff) > tol).any():
        if num_iters > max_iters:
            raise Warning(
                f"Exceeded {max_iters} iterations without sigma changing by less than "
                f"{tol:.1e}"
            )

        cos_2sigmam = torch.cos(2 * sigma1 + sigma)
        delta_sigma = (
            B
            * torch.sin(sigma)
            * (
                cos_2sigmam
                + (1 / 4)
                * B
                * (
                    torch.cos(sigma) * (-1 + 2 * cos_2sigmam**2)
                    - (1 / 6)
                    * B
                    * cos_2sigmam
                    * (-3 + 4 * torch.sin(sigma) ** 2)
                    * (-3 + 4 * cos_2sigmam**2)
                )
            )
        )
        sigma_i = s / (WGS_84_B * A) + delta_sigma
        sigma_diff = sigma_i - sigma
        sigma = sigma_i
        num_iters += 1

    lat2 = torch.atan2(
        torch.sin(U1) * torch.cos(sigma)
        + torch.cos(U1) * torch.sin(sigma) * torch.cos(alpha1),
        (1 - WGS_84_F)
        * torch.sqrt(
            sin_alpha**2
            + (
                torch.sin(U1) * torch.sin(sigma)
                - torch.cos(U1) * torch.cos(sigma) * torch.cos(alpha1)
            )
            ** 2
        ),
    )
    lambd = torch.atan2(
        torch.sin(sigma) * torch.sin(alpha1),
        torch.cos(U1) * torch.cos(sigma)
        - torch.sin(U1) * torch.sin(sigma) * torch.cos(alpha1),
    )
    C = (
        (WGS_84_F / 16)
        * (1 - sin_alpha**2)
        * (4 + WGS_84_F * (4 - 3 * (1 - sin_alpha**2)))
    )
    L = lambd - (1 - C) * WGS_84_F * sin_alpha * (
        sigma
        + C
        * torch.sin(sigma)
        * (cos_2sigmam + C * torch.cos(sigma) * (-1 + 2 * cos_2sigmam**2))
    )
    lon2 = L + lon1
    alpha2 = torch.atan2(
        sin_alpha,
        -torch.sin(U1) * torch.sin(sigma)
        + torch.cos(U1) * torch.cos(sigma) * torch.cos(alpha1),
    )

    lat2, lon2 = lat2 * 180 / torch.pi, lon2 * 180 / torch.pi
    if isinstance(latlon1, tuple):
        latlon2 = (lat2, lon2)
    else:
        latlon2 = torch.stack([lat2, lon2])
    return latlon2, alpha2
