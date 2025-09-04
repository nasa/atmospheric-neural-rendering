"""A set of geospatial functions for conversions between spherical Earth horizontal,
spherical Earth Cartesian, and normalized Cartesian scene coordinates.
These functions use torch tensors and are 'device-aware' because single-threaded
geospatial operations on CPU are too slow to keep up with many neural rendering models.
"""

import torch

from atmonr.geospatial.wgs_84 import WGS_84_A, WGS_84_B


EARTH_RADIUS = 6.378e6  # meters


def wgs_84_to_spherical(xyz):
    z = xyz[..., 2] * WGS_84_A / WGS_84_B
    xyz_spherical = torch.cat([xyz[..., :2], z[..., None]], dim=-1)
    return xyz_spherical * EARTH_RADIUS / WGS_84_A


def spherical_to_wgs84(xyz):
    xyz_wgs84 = xyz * WGS_84_A / EARTH_RADIUS
    xyz_wgs84[..., 2] *= WGS_84_B / WGS_84_A
    return xyz_wgs84


def stretch_above_sea_level(xyz: torch.Tensor, stretch: float):
    radii = torch.linalg.norm(xyz, dim=-1)
    above_surf = radii > EARTH_RADIUS
    rad_stretch = radii.clone()
    rad_stretch[above_surf] = (
        radii[above_surf] - EARTH_RADIUS
    ) * stretch + EARTH_RADIUS
    xyz_stretch = torch.clone(xyz)
    xyz_stretch[above_surf] *= (rad_stretch[above_surf] / radii[above_surf])[:, None]
    return xyz_stretch
