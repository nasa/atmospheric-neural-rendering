from pathlib import Path

import netCDF4
import numpy as np

try:
    import openvdb as vdb  # type: ignore
except ImportError:
    try:
        import pyopenvdb as vdb  # type: ignore
    except ImportError:
        vdb = None
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import warnings

from atmonr.datasets.harp2 import HARP2Dataset, download_harp2_file
from atmonr.geospatial.spherical import (
    wgs_84_to_spherical,
    spherical_to_wgs84,
    stretch_above_sea_level,
)
from atmonr.geospatial.wgs_84 import (
    cartesian_to_horizontal,
    horizontal_to_cartesian,
    vincenty_distance,
    vincenty_point_along_geodesic,
)
from atmonr.graphics_utils import voxel_traversal


_CHUNK_SIZE = int(5e4)
DEM_PATH = "data/ETOPO1_ocssw.nc"


class HARP2ExtractDataset(Dataset):
    """Dataset for getting the extinction coefficient field from a HARP2Dataset.

    This class is abstract and should not be directly instantiated.
    """

    def __init__(self, dataset: HARP2Dataset) -> None:
        if type(self) is HARP2ExtractDataset:
            raise NotImplementedError
        super().__init__()
        self.dataset = dataset
        self.device = dataset.lat.device
        self.shp = (0, 0)
        self.xyz = torch.zeros(0, device=self.device)
        self.idx = torch.zeros(0, dtype=torch.int32, device=self.device)

    def __getitem__(self, idx: int | torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "xyz": self.xyz[idx],
            "idx": self.idx[idx],
        }

    def __getbatch__(self, idx: torch.Tensor) -> dict[str, torch.Tensor]:
        return self[idx]

    def dump(
        self,
        output_filepath: Path,
        sigma: torch.Tensor,
    ) -> None:
        raise NotImplementedError


class _HARP2LocalExtractDataset(HARP2ExtractDataset):
    """HARP2ExtractDataset for local, i.e. non-global, grids.

    This class is abstract and should not be instantiated.
    """

    def __init__(
        self,
        dataset: HARP2Dataset,
        alt_step: float,
        min_alt: float | None = None,
        max_alt: float | None = None,
        *args,
        **kwargs,
    ) -> None:
        if type(self) is _HARP2LocalExtractDataset:
            raise NotImplementedError
        super().__init__(dataset)
        self.alt_step = alt_step
        self.min_alt = -self.dataset.subsurface_depth if min_alt is None else min_alt
        self.max_alt = self.dataset.ray_origin_height if max_alt is None else max_alt
        self.sample_alt = torch.arange(
            self.min_alt,
            self.max_alt + self.alt_step / 2,
            self.alt_step,
        ).to(self.device)

    def dump(
        self,
        output_filepath: Path,
        sigma: torch.Tensor,
    ) -> None:
        """Dump this dataset to a netCDF file.

        Args:
            output_filepath: Path to a .netCDF file in which to dump the data.
            sigma: The 3D field of the extinction coefficient.
        """

        _extract_to_netCDF(output_filepath, self, sigma)


class HARP2L1CExtractDataset(_HARP2LocalExtractDataset):
    """Implementation of HARP2LocalExtractDataset for the L1C grid. This loads the level
    1C data (downloading it if necessary) corresponding to the level 1B file in the
    provided HARP2Dataset, and uses the level 1C latitude and longitude to define the
    horizontal spacing of the grid. Using the level 1B latitude and longitude is not
    ideal for visualization due to their angle-dependence and relatively non-uniform
    spacing. The vertical spacing of the grid is defined by the user.
    """

    def __init__(
        self,
        dataset: HARP2Dataset,
        alt_step: float,
        min_alt: float | None = None,
        max_alt: float | None = None,
        *args,
        **kwargs,
    ) -> None:
        """Initialize a HARP2L1CExtractDataset.

        Args:
            alt_step: Vertical spacing between voxels, in meters.
            min_alt: Minimum altitude above sea-level of the voxel grid, in meters.
            max_alt: Maximum altitude above sea-level of the voxel grid, in meters.
        """
        super().__init__(dataset, alt_step, min_alt, max_alt)

        # get the corresponding l1c filename download it if it's missing
        sensor, timestamp, _, version, _ = self.dataset.filename.split(".")
        l1c_filename = f"{sensor}.{timestamp}.L1C.{version}.5km.nc"
        l1c_path = Path("data/HARP2_L1C") / l1c_filename
        if not l1c_path.exists():
            download_harp2_file(l1c_filename, l1c_path.parent, "L1C")

        # load the netCDF4 file
        self.l1c_nc_data = netCDF4.Dataset(l1c_path)
        self.shp = self.l1c_nc_data["geolocation_data/latitude"].shape

        def _parse_field(
            field: netCDF4.Variable,
        ) -> torch.Tensor:
            """Read a field in HARP2 L1C data, performing the following transformations:
            1) fill invalid values with nan
            2) flip the y-axis so North is at the top of the image
            3) convert to torch tensor
            """
            arr = np.ascontiguousarray(field[:].filled(fill_value=np.nan)[::-1])
            return torch.from_numpy(arr).to(self.device)

        # read the HARP2 L1C fields, filling invalid with nan and flipping north
        lat = _parse_field(self.l1c_nc_data["geolocation_data/latitude"])
        lon = _parse_field(self.l1c_nc_data["geolocation_data/longitude"])
        self.height = _parse_field(self.l1c_nc_data["geolocation_data/height"])

        self.lat = lat[:, :, None].repeat((1, 1, self.sample_alt.shape[0]))
        self.lon = lon[:, :, None].repeat((1, 1, self.sample_alt.shape[0]))
        # self.alt = self.height[:, :, None] + self.sample_alt[None, None]
        alt = self.sample_alt[None, None].repeat(
            self.lat.shape[0], self.lat.shape[1], 1
        )

        xyz = torch.stack(
            list(
                horizontal_to_cartesian(
                    self.lat.double(), self.lon.double(), alt.double()
                )
            ),
            dim=-1,
        )
        self.xyz = xyz.view(-1, 3)
        self.idx = torch.arange(self.xyz.shape[0], dtype=torch.int32)


class HARP2VoxelGridExtractDataset(_HARP2LocalExtractDataset):
    """Implementation of HARP2LocalExtractDataset for a user-defined voxel grid. This
    grid attempts to keep horizontal spacing as uniform as possible, despite Earth
    curvature, by using the Vincenty distance.
    """

    def __init__(
        self,
        dataset: HARP2Dataset,
        horizontal_step: float,
        alt_step: float,
        min_alt: float | None = None,
        max_alt: float | None = None,
        *args,
        **kwargs,
    ) -> None:
        """Initialize a HARP2VoxelGridExtractDataset.

        Args:
            horizontal_step: Horizontal spacing between voxels, in meters.
            alt_step: Vertical spacing between voxels, in meters.
            min_alt: Minimum altitude above sea-level of the voxel grid, in meters.
            max_alt: Maximum altitude above sea-level of the voxel grid, in meters.
        """
        super().__init__(dataset, alt_step, min_alt, max_alt)

        self.horizontal_step = horizontal_step
        self.alt_step = alt_step
        self.min_alt = -self.dataset.subsurface_depth if min_alt is None else min_alt
        self.max_alt = self.dataset.ray_origin_height if max_alt is None else max_alt

        # get bounding lat/lon, making 2 assumptions:
        # 1) the lat/lon arrays are ordered so u is decreasing in latitude and v is increasing in longitude
        # 2) the corners of the lat/lon arrays have at least one valid value
        lat_img = self.dataset.lat.view(list(self.dataset.img_shp) + [90])
        lon_img = self.dataset.lon.view(list(self.dataset.img_shp) + [90])

        # check assumption 1
        assert torch.nanmean(lat_img[-1, 0] - lat_img[0, 0]) < 0
        lon_mean_diff = torch.nanmean(lon_img[0, -1] - lon_img[0, 0]) % 360
        assert lon_mean_diff > 0 and lon_mean_diff < 180
        # check assumption 2
        for i, j in [(0, 0), (0, -1), (-1, 0), (-1, -1)]:
            assert not lat_img[i, j].isnan().all()
            assert not lon_img[i, j].isnan().all()

        # shift longitude to avoid 180 longitude issues
        lon_img_mean = torch.nanmean(lon_img)
        lon_img_shifted = lon_img - lon_img_mean

        def _fix_lon(lon):
            return (lon + 180) % 360 - 180

        # get the four corners
        def _nanmax(x):
            return x[~x.isnan()].max()

        def _nanmin(x):
            return x[~x.isnan()].min()

        topleft = (
            _nanmax(lat_img[0, 0]),
            _fix_lon(_nanmin(lon_img_shifted[0, 0]) + lon_img_mean),
        )
        botleft = (
            _nanmin(lat_img[-1, 0]),
            _fix_lon(_nanmin(lon_img_shifted[-1, 0]) + lon_img_mean),
        )
        topright = (
            _nanmax(lat_img[0, -1]),
            _fix_lon(_nanmax(lon_img_shifted[0, -1]) + lon_img_mean),
        )
        botright = (
            _nanmin(lat_img[-1, -1]),
            _fix_lon(_nanmax(lon_img_shifted[-1, -1]) + lon_img_mean),
        )

        def _get_midpoint(latlon1, latlon2):
            s, alpha1, _ = vincenty_distance(latlon1, latlon2)
            midpoint, _ = vincenty_point_along_geodesic(latlon1, alpha1, s / 2)
            return midpoint

        # get midpoints of top, left, right, and bottom edges of granule
        topmid = _get_midpoint(topleft, topright)
        leftmid = _get_midpoint(topleft, botleft)
        rightmid = _get_midpoint(topright, botright)
        botmid = _get_midpoint(botleft, botright)

        # get distance of cross lines
        dist_lr, _, _ = vincenty_distance(leftmid, rightmid)
        dist_tb, _, _ = vincenty_distance(topmid, botmid)

        # get horizontal shape of voxel grid
        img_shp = (
            int(dist_tb // self.horizontal_step),
            int(dist_lr // self.horizontal_step),
        )
        pad = dist_tb % self.horizontal_step, dist_lr % self.horizontal_step

        # get ratios of samples to take
        samples_tb = (
            torch.linspace(0, dist_tb - pad[0], img_shp[0]).to(self.device) + pad[0] / 2
        ) / dist_tb
        samples_lr = (
            torch.linspace(0, dist_lr - pad[1], img_shp[1]).to(self.device) + pad[1] / 2
        ) / dist_lr

        # get samples along top and bottom of granule
        along_top_distance, top_azi, _ = vincenty_distance(topleft, topright)
        along_bot_distance, bot_azi, _ = vincenty_distance(botleft, botright)
        samples_along_top, _ = vincenty_point_along_geodesic(
            torch.stack(topleft),
            torch.FloatTensor([top_azi]).to(self.device),
            samples_lr * along_top_distance,
        )
        samples_along_bot, _ = vincenty_point_along_geodesic(
            torch.stack(botleft),
            torch.FloatTensor([bot_azi]).to(self.device),
            samples_lr * along_bot_distance,
        )

        # get samples between those samples, to get a grid
        column_distances, column_azi, _ = vincenty_distance(
            samples_along_top, samples_along_bot
        )
        assert isinstance(samples_along_top, torch.Tensor)
        assert isinstance(column_distances, torch.Tensor)
        (sample_lat, sample_lon), _ = vincenty_point_along_geodesic(
            samples_along_top[:, None],
            column_azi[None],
            (samples_tb[:, None] * column_distances[None]),
        )

        self.height = self._interp_dem_height(
            netCDF4.Dataset(DEM_PATH), sample_lat, sample_lon
        )

        self.lat = sample_lat[:, :, None].repeat((1, 1, self.sample_alt.shape[0]))
        self.lon = sample_lon[:, :, None].repeat((1, 1, self.sample_alt.shape[0]))
        alt = self.sample_alt[None, None].repeat(*(list(self.lat.shape)[:2] + [1]))

        xyz = torch.stack(
            list(
                horizontal_to_cartesian(
                    self.lat.double(), self.lon.double(), alt.double()
                )
            ),
            dim=-1,
        )

        self.shp = self.lat.shape
        self.xyz = xyz.view(-1, 3)
        self.idx = torch.arange(self.xyz.shape[0], dtype=torch.int32)

    def _interp_dem_height(
        self,
        dem_dataset: netCDF4.Dataset,
        sample_lat: torch.Tensor,
        sample_lon: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate the Digital Elevation Map Dataset to user-provided coordinates.

        Args:
            dem_dataset: Digital Elevation Map dataset "ETOPO1_ocssw.nc".
            sample_lat: Latitudes at which to interpolate the DEM.
            sample_lon: Longitudes at which to interpolate the DEM.

        Returns:
            interp_height: DEM height interpolated to the provided coordinates.
        """
        assert sample_lat.shape == sample_lon.shape

        shp = sample_lat.shape

        dem_upper_lat = dem_dataset.upper_lat.item()
        dem_left_lon = dem_dataset.left_lon.item()

        dem_lat_res = dem_dataset.geospatial_lat_resolution.item()
        dem_lon_res = dem_dataset.geospatial_lon_resolution.item()

        sample_loc_dem_lat = (dem_upper_lat - sample_lat) / dem_lat_res
        sample_loc_dem_lon = (sample_lon - dem_left_lon) / dem_lon_res

        idx_lat = (sample_loc_dem_lat // 1).int()
        idx_lon = (sample_loc_dem_lon // 1).int()
        rem_lat = (sample_loc_dem_lat % 1).flatten()
        rem_lon = (sample_loc_dem_lon % 1).flatten()

        # make sure top left indices are in bounds
        idx_lat = torch.clamp(
            idx_lat, 0, dem_dataset.dimensions["lat"].size - 2
        ).flatten()
        idx_lon = torch.clamp(
            idx_lon, 0, dem_dataset.dimensions["lon"].size - 2
        ).flatten()

        # read only a subset of the dem height from file, for speed
        dem_height = dem_dataset["height"][
            idx_lat.min() : idx_lat.max() + 2, idx_lon.min() : idx_lon.max() + 2
        ]
        dem_water_surface_height = dem_dataset["water_surface_height"][
            idx_lat.min() : idx_lat.max() + 2, idx_lon.min() : idx_lon.max() + 2
        ]
        dem_height = torch.from_numpy(dem_height).cuda()
        dem_water_surface_height = torch.from_numpy(dem_water_surface_height).cuda()
        dem_height = torch.maximum(dem_height, dem_water_surface_height)
        # update the indices
        idx_lat, idx_lon = idx_lat - idx_lat.min(), idx_lon - idx_lon.min()

        # get interpolation corners and weights
        corners = torch.stack(
            [
                dem_height[idx_lat, idx_lon],  # top left
                dem_height[idx_lat, idx_lon + 1],  # top right
                dem_height[idx_lat + 1, idx_lon],  # bottom left
                dem_height[idx_lat + 1, idx_lon + 1],  # bottom right
            ]
        )
        weights = torch.stack(
            [
                (1 - rem_lat) * (1 - rem_lon),
                (1 - rem_lat) * rem_lon,
                (rem_lat * (1 - rem_lon)),
                rem_lat * rem_lon,
            ]
        )

        # multiply and sum
        interp_height = (corners * weights).sum(dim=0).view(shp)
        interp_height = torch.clamp(interp_height, min=0)
        return interp_height


def _extract_to_netCDF(
    output_filepath: Path,
    extract_dataset: _HARP2LocalExtractDataset,
    sigma: torch.Tensor,
) -> None:
    """Write a L1C or voxelgrid extract to a netCDF file.

    Args:
        output_filepath: The path of the output netCDF file.
        extract_dataset: The extract dataset to write to file.
        sigma: 3D field of the extinction coefficient as a torch tensor.
    """
    assert output_filepath.suffix == ".nc"
    assert isinstance(
        extract_dataset, HARP2L1CExtractDataset | HARP2VoxelGridExtractDataset
    )

    num_bands = sigma.shape[-1]
    sigma = sigma.view(
        list(extract_dataset.shp[:2]) + [extract_dataset.sample_alt.shape[0], num_bands]
    )
    ncfile = netCDF4.Dataset(output_filepath, mode="w")

    # dimensions
    ncfile.createDimension("bins_along_track", extract_dataset.lat.shape[0])
    ncfile.createDimension("bins_across_track", extract_dataset.lat.shape[1])
    ncfile.createDimension("bins_vertical", extract_dataset.sample_alt.shape[0])
    ncfile.createDimension("number_of_bands", num_bands)
    ncfile.createDimension("number_of_views", 90)

    # attributes
    ncfile.title = "PACE HARP2 Neural Rendering Volumetric Data"
    ncfile.input_l1b_product_name = extract_dataset.dataset.nc_data.product_name
    if isinstance(extract_dataset, HARP2L1CExtractDataset):
        ncfile.input_l1c_product_name = extract_dataset.l1c_nc_data.product_name
    ncfile.neural_rendering_scene_scale = extract_dataset.dataset.scale
    ncfile.neural_rendering_scene_offset_x = extract_dataset.dataset.offset[0].item()
    ncfile.neural_rendering_scene_offset_y = extract_dataset.dataset.offset[1].item()
    ncfile.neural_rendering_scene_offset_z = extract_dataset.dataset.offset[2].item()

    # Some geolocation attributes
    lat = ncfile.createVariable(
        "latitude",
        np.float32,
        ("bins_along_track", "bins_across_track"),
        fill_value=-32767,
    )
    lon = ncfile.createVariable(
        "longitude",
        np.float32,
        ("bins_along_track", "bins_across_track"),
        fill_value=-32767,
    )
    height = ncfile.createVariable(
        "height",
        np.float32,
        ("bins_along_track", "bins_across_track"),
        fill_value=-32767,
    )
    solar_zen, solar_azi = None, None
    if isinstance(extract_dataset, HARP2L1CExtractDataset):
        solar_zen = ncfile.createVariable(
            "solar_zenith_angle",
            np.float32,
            ("bins_along_track", "bins_across_track", "number_of_views"),
            fill_value=-32767,
        )
        solar_azi = ncfile.createVariable(
            "solar_azimuth_angle",
            np.float32,
            ("bins_along_track", "bins_across_track", "number_of_views"),
            fill_value=-32767,
        )

    # if in L1C mode, copy over the geolocation data
    def _copy_nc_var(src, dst):
        dst.setncatts({k: v for k, v in src.__dict__.items() if k != "_FillValue"})
        dst[:] = src[:]

    if isinstance(extract_dataset, HARP2L1CExtractDataset):
        _copy_nc_var(extract_dataset.l1c_nc_data["geolocation_data/latitude"], lat)
        _copy_nc_var(extract_dataset.l1c_nc_data["geolocation_data/longitude"], lon)
        _copy_nc_var(extract_dataset.l1c_nc_data["geolocation_data/height"], height)
        _copy_nc_var(
            extract_dataset.l1c_nc_data["geolocation_data/solar_zenith_angle"],
            solar_zen,
        )
        _copy_nc_var(
            extract_dataset.l1c_nc_data["geolocation_data/solar_azimuth_angle"],
            solar_azi,
        )
    # otherwise, use our own
    else:
        lat.long_name = "Latitude of bin locations"
        lat.units = "degrees_north"
        lat.valid_min = -90.0
        lat.valid_max = 90.0
        lat[:] = extract_dataset.lat[..., 0].cpu().numpy()
        lon.long_name = "Longitude of bin locations"
        lon.units = "degrees_east"
        lon.valid_min = -180.0
        lon.valid_max = 180.0
        lon[:] = extract_dataset.lon[..., 0].cpu().numpy()
        height.long_name = "Altitude at bin locations"
        height.units = "meters"
        height.valid_min = -1000
        height.valid_max = 10000
        height[:] = extract_dataset.height.cpu().numpy()

    # altitude
    nc_sample_alt = ncfile.createVariable(
        "altitude", np.float32, ("bins_vertical",), fill_value=-32767
    )
    nc_sample_alt.units = "meters"
    nc_sample_alt.long_name = "Altitude above surface"
    nc_sample_alt[:] = extract_dataset.sample_alt.cpu().numpy()

    # extinction coefficient
    nc_sigma = ncfile.createVariable(
        "extinction_coefficient",
        np.float32,
        (
            "bins_along_track",
            "bins_across_track",
            "bins_vertical",
            "number_of_bands",
        ),
        fill_value=-32767,
    )
    nc_sigma.units = "m^-1"
    nc_sigma.long_name = "Extinction coefficient"
    nc_sigma.valid_min = 0
    nc_sigma[:] = sigma.cpu().numpy()

    # WGS-84 Cartesian XYZ
    xyz_vg = (
        extract_dataset.xyz.view(
            list(extract_dataset.shp[:2]) + [extract_dataset.sample_alt.shape[0], 3]
        )
        .cpu()
        .numpy()
    )
    nc_x = ncfile.createVariable(
        "x_wgs84",
        np.float32,
        ("bins_along_track", "bins_across_track", "bins_vertical"),
    )
    nc_y = ncfile.createVariable(
        "y_wgs84",
        np.float32,
        ("bins_along_track", "bins_across_track", "bins_vertical"),
    )
    nc_z = ncfile.createVariable(
        "z_wgs84",
        np.float32,
        ("bins_along_track", "bins_across_track", "bins_vertical"),
    )
    nc_x.units = "meters"
    nc_y.units = "meters"
    nc_z.units = "meters"
    nc_x.long_name = "X coordinate in WGS-84 cartesian (EPSG:4978)"
    nc_y.long_name = "Y coordinate in WGS-84 cartesian (EPSG:4978)"
    nc_z.long_name = "Z coordinate in WGS-84 cartesian (EPSG:4978)"
    nc_x[:] = xyz_vg[..., 0]
    nc_y[:] = xyz_vg[..., 1]
    nc_z[:] = xyz_vg[..., 2]

    ncfile.close()


class HARP2GlobalGridExtractDataset(HARP2ExtractDataset):
    """Implementation of HARP2ExtractDataset for a global voxel grid, for large-scale
    visualization purposes. This dataset uses a spherical Earth reference frame and a
    voxel grid with a user-provided scale and grid resolution. This mode also allows the
    stretching of the atmosphere to make it easier to see vertical variability in large
    scale visualizations.
    """

    def __init__(
        self,
        dataset: HARP2Dataset,
        scale: float,
        grid_res: float,
        vstretch: float | None = None,
        *args,
        **kwargs,
    ) -> None:
        """Initialize a HARP2GlobalGridExtractDataset.

        Args:
            scale: Scale of the voxel grid.
            grid_res: The voxel grid resolution in the provided scale.
            vstretch: Factor by which to stretch the above-surface points.
        """
        super().__init__(dataset)

        if vstretch is None:
            vstretch = 1
        assert vstretch >= 1

        self.scale = scale
        self.grid_res = grid_res
        self.vstretch = vstretch

        # convert from WGS-84 to spherical
        ray_origin = wgs_84_to_spherical(self.dataset.ray_origin)
        ray_dest = (
            self.dataset.ray_origin
            + self.dataset.ray_dir * self.dataset.ray_len[:, None]
        )
        ray_dest = wgs_84_to_spherical(ray_dest)

        # stretch all points above sea level
        ray_origin = stretch_above_sea_level(ray_origin, self.vstretch)
        ray_dest = stretch_above_sea_level(ray_dest, self.vstretch)

        # convert from spherical to continuous voxel grid indices
        ray_origin *= self.scale / self.grid_res
        ray_dest *= self.scale / self.grid_res

        # voxel traversal in chunks to avoid running out of GPU memory
        self.voxels = torch.zeros((0, 3)).to(device=self.device)
        for i in tqdm(
            range(ray_origin.shape[0] // _CHUNK_SIZE + 1),
            desc="Traversing voxels",
        ):
            start = min(ray_origin.shape[0], i * _CHUNK_SIZE)
            end = min(ray_origin.shape[0], start + _CHUNK_SIZE)
            if start == end:
                continue
            self.voxels = torch.cat(
                [
                    self.voxels,
                    voxel_traversal(
                        ray_origin[start:end],
                        ray_dest[start:end],
                        unique_only=False,
                    ),
                ],
                dim=0,
            )
            # have to do this every chunk, or will run out of GPU memory
            self.voxels = torch.unique(self.voxels, dim=0, sorted=False)
        del (ray_origin, ray_dest)

        # voxel centers in meters
        self.xyz = (self.voxels.float() + 0.5) * (self.grid_res / scale)
        # un-stretch using the reciprocal
        self.xyz = stretch_above_sea_level(self.xyz, 1 / self.vstretch)
        # convert back to WGS-84
        self.xyz = spherical_to_wgs84(self.xyz)
        # cull any voxels with centers above ray_origin_height or below surface
        _, _, alt = cartesian_to_horizontal(
            self.xyz[..., 0],
            self.xyz[..., 1],
            self.xyz[..., 2],
        )
        cull = alt <= 0
        if self.dataset.ray_origin_height is not None:
            cull += alt > self.dataset.ray_origin_height
        self.xyz, self.voxels = self.xyz[~cull], self.voxels[~cull]
        self.idx = torch.arange(self.xyz.shape[0], dtype=torch.int32)

    def dump(
        self,
        output_filepath: Path,
        sigma: torch.Tensor,
    ) -> None:
        """Dump this dataset to an OpenVDB file, unless it is not installed, in which
        case, write the intermediate outputs to numpy files, which can be written to
        OpenVDB elsewhere. The second option is useful if you are unable to install the
        OpenVDB Python bindings in the same environment as your training.

        Args:
            output_filepath: Path to a .vdb file in which to dump the data.
            sigma: The 3D field of the extinction coefficient.
        """
        if vdb is None:
            voxel_filepath = Path("voxels.npy")
            sigma_filepath = Path("sigma.npy")
            warnings.warn(
                "Unable to import OpenVDB Python bindings, exporting to "
                f"{voxel_filepath} and {sigma_filepath} instead."
            )
            if voxel_filepath.exists():
                raise FileExistsError
            if sigma_filepath.exists():
                raise FileExistsError
            np.save(
                voxel_filepath, self.voxels.detach().cpu().numpy(), allow_pickle=False
            )
            np.save(sigma_filepath, sigma.detach().cpu().numpy(), allow_pickle=False)
            return
        assert output_filepath.suffix == ".vdb"

        grid = vdb.FloatGrid()
        for i in tqdm(range(sigma.shape[0])):
            grid.copyFromArray(sigma[i, None, None, None], ijk=self.voxels[i])

        grid.transform = vdb.createLinearTransform(voxelSize=self.grid_res)
        grid.name = "density"
        grid.saveFloatAsHalf = True
        grid.vectorType = "invariant"

        vdb.write(str(output_filepath), grids=[grid])
