from pathlib import Path

import netCDF4
import numpy as np
import torch
from torch.utils.data import Dataset

from atmonr.datasets.harp2 import HARP2Dataset, download_harp2_file
from atmonr.geometry import (
    horizontal_to_cartesian,
    vincenty_distance,
    vincenty_point_along_geodesic,
)


DEM_PATH = "data/ETOPO1_ocssw.nc"


class HARP2ExtractDataset(Dataset):
    """ExtractDataset for getting the extinction coefficient field from a HARP2Dataset.

    Defines a 3D grid over a HARP2Dataset, using the provided mode, horizontal_step, and
    sample_alt, and allows iteration over this grid.

    The `mode` should be "native" or "voxelgrid". In "native" mode, this ExtractDataset
    loads the level 1C data (downloading it if necessary) corresponding to the level 1B
    file in the provided HARP2Dataset, and uses the level 1C latitude and longitude to
    define the grid. Using the level 1B latitude and longitude is not practical due to
    their angle-dependence and relatively non-uniform spacing. In "voxelgrid" mode, the
    provided horizontal_step and sample_alt are used to define a grid of nearly uniform
    voxel-like cells.
    """

    def __init__(
        self,
        mode: str,
        dataset: HARP2Dataset,
        horizontal_step: float,
        sample_alt: torch.Tensor,
    ) -> None:
        """Initialize a HARP2ExtractDataset.

        Args:
            mode: Extraction mode, either "native" or "voxelgrid".
            dataset: The HARP2Dataset from which to extract the extinction coefficient.
            horizontal_step: Horizontal spacing in meters between cells. Only used in
                "voxelgrid" mode.
            sample_alt: Sample altitudes defining the vertical spacing between cells.
        """
        super().__init__()

        assert mode in ["native", "voxelgrid"]
        self.mode = mode
        self.dataset = dataset
        self.horizontal_step = horizontal_step
        self.sample_alt = sample_alt

        self.device = dataset.lat.device

        if self.mode == "native":
            self._init_native_grid()
        else:
            self._init_voxel_grid()
        self.idx = torch.arange(self.xyz.shape[0], dtype=torch.int32)

    def _init_native_grid(self) -> None:
        """Initialize the native mode grid, defined by the latitude and longitude in the
        level 1C data and a user-configured sample altitudes."""
        # get the corresponding L1C filename download it if it's missing
        sensor, timestamp, _, version, _ = self.dataset.filename.split(".")
        l1c_filename = f"{sensor}.{timestamp}.L1C.{version}.5km.nc"
        l1c_path = Path("data/HARP2_L1C") / l1c_filename
        if not l1c_path.exists():
            download_harp2_file(l1c_filename, l1c_path.parent, "L1C")

        # load the netCDF4 file
        self.l1c_nc_data = netCDF4._netCDF4.Dataset(l1c_path)
        self.shp = self.l1c_nc_data["geolocation_data/latitude"].shape

        def _parse_field(
            field: netCDF4._netCDF4.Variable,
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

    def _init_voxel_grid(self) -> None:
        """Initialize a voxelgrid mode grid, defined by the user-configured horizontal
        spacing and sample altitudes."""
        # get bounding lat/lon, making 2 assumptions:
        # 1) the lat/lon arrays are ordered so u is decreasing in latitude and v is increasing in longitude
        # 2) the corners of the lat/lon arrays have at least one valid value
        lat_img = self.dataset.lat.view(list(self.dataset.img_shp) + [90])
        lon_img = self.dataset.lon.view(list(self.dataset.img_shp) + [90])

        # check assumption 1
        assert torch.nanmean(lat_img[-1, 0] - lat_img[0, 0]) < 0
        assert torch.nanmean(lon_img[0, -1] - lon_img[0, 0]) > 0
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
            netCDF4._netCDF4.Dataset(DEM_PATH), sample_lat, sample_lon
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

    def _interp_dem_height(
        self,
        dem_dataset: netCDF4._netCDF4.Dataset,
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

    def __getitem__(self, idx: int | torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "xyz": self.xyz[idx],
            "idx": self.idx[idx],
        }

    def __getbatch__(self, idx: torch.Tensor) -> dict[str, torch.Tensor]:
        return self[idx]

    def write_netcdf(self, output_filepath: Path, sigma: torch.Tensor) -> None:
        """Write the extracted data to a netCDF file at the provided location.

        Args:
            output_filepath: Path where the extracted netCDF should be saved.
            sigma: 3D field of the Extinction coefficient.
        """
        num_bands = sigma.shape[-1]
        sigma = sigma.view(list(self.shp[:2]) + [self.sample_alt.shape[0], num_bands])
        ncfile = netCDF4._netCDF4.Dataset(output_filepath, mode="w")

        # dimensions
        ncfile.createDimension("bins_along_track", self.lat.shape[0])
        ncfile.createDimension("bins_across_track", self.lat.shape[1])
        ncfile.createDimension("bins_vertical", self.sample_alt.shape[0])
        ncfile.createDimension("number_of_bands", num_bands)
        ncfile.createDimension("number_of_views", 90)

        # attributes
        ncfile.title = "PACE HARP2 Neural Rendering Volumetric Data"
        ncfile.input_l1b_product_name = self.dataset.nc_data.product_name
        if self.mode == "native":
            ncfile.input_l1c_product_name = self.l1c_nc_data.product_name
        ncfile.neural_rendering_scene_scale = self.dataset.scale
        ncfile.neural_rendering_scene_offset_x = self.dataset.offset[0].item()
        ncfile.neural_rendering_scene_offset_y = self.dataset.offset[1].item()
        ncfile.neural_rendering_scene_offset_z = self.dataset.offset[2].item()
        ncfile.extract_mode = self.mode

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
        if self.mode == "native":
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

        # if in native mode, copy over the geolocation data
        def _copy_nc_var(src, dst):
            dst.setncatts({k: v for k, v in src.__dict__.items() if k != "_FillValue"})
            dst[:] = src[:]

        if self.mode == "native":
            _copy_nc_var(self.l1c_nc_data["geolocation_data/latitude"], lat)
            _copy_nc_var(self.l1c_nc_data["geolocation_data/longitude"], lon)
            _copy_nc_var(self.l1c_nc_data["geolocation_data/height"], height)
            _copy_nc_var(
                self.l1c_nc_data["geolocation_data/solar_zenith_angle"], solar_zen
            )
            _copy_nc_var(
                self.l1c_nc_data["geolocation_data/solar_azimuth_angle"], solar_azi
            )
        # otherwise, use our own
        else:
            lat.long_name = "Latitude of bin locations"
            lat.units = "degrees_north"
            lat.valid_min = -90.0
            lat.valid_max = 90.0
            lat[:] = self.lat[..., 0].cpu().numpy()
            lon.long_name = "Longitude of bin locations"
            lon.units = "degrees_east"
            lon.valid_min = -180.0
            lon.valid_max = 180.0
            lon[:] = self.lon[..., 0].cpu().numpy()
            height.long_name = "Altitude at bin locations"
            height.units = "meters"
            height.valid_min = -1000
            height.valid_max = 10000
            height[:] = self.height.cpu().numpy()

        # altitude
        nc_sample_alt = ncfile.createVariable(
            "altitude", np.float32, ("bins_vertical",), fill_value=-32767
        )
        nc_sample_alt.units = "meters"
        nc_sample_alt.long_name = "Altitude above surface"
        nc_sample_alt[:] = self.sample_alt.cpu().numpy()

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
            self.xyz.view(list(self.shp[:2]) + [self.sample_alt.shape[0], 3])
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
