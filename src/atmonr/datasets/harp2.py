from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

import earthaccess
import netCDF4
import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from torch.utils.data import Dataset
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)

from atmonr.geometry import (
    cartesian_to_horizontal,
    filter_rays,
    get_rays,
    normalize_rays,
)
from atmonr.progress_tracker import ProgressTracker


class HARP2Dataset(Dataset):
    """Dataset for the Hyper-Angular Rainbow Polarimeter 2 (HARP2) level 1B product.

    Allows loading and iteration over HARP2 data formatted for neural rendering.
    """

    def __init__(
        self,
        filename: str,
        ray_origin_height: float = 20000.0,
        subsurface_depth: float = 1000.0,
    ) -> None:
        """Initialize a HARP2Dataset.

        Args:
            filename: Filename corresponding to a HARP2 file.
            ray_origin_height: Altitude at which to construct ray origins, i.e. the
                'top' of a scene.
        """
        super().__init__()
        self.filename = filename
        self.ray_origin_height = ray_origin_height
        self.subsurface_depth = subsurface_depth
        self.local_path = Path("data/HARP2") / filename

        # if the file does not exist, go get it using earthaccess
        if not self.local_path.exists():
            download_harp2_file(self.filename, self.local_path.parent, "L1B")

        # load the netCDF4 file
        self.nc_data = netCDF4.Dataset(self.local_path)

        # used to reorder the channels, as HARP2 data is in GRNB order but we want BGRN
        self.band_order = torch.cat(
            [
                torch.arange(80, 90),
                torch.arange(10),
                torch.arange(10, 70),
                torch.arange(70, 80),
            ]
        ).cuda()

        self._parse_ray_data()
        self._select_best_slice()

        # convert from image-like data to a flattened array of rays
        self.ray_origin, self.ray_dir, self.ray_len = get_rays(
            self.lat,
            self.lon,
            self.alt,
            self.thetav,
            self.phiv,
            ray_origin_height=self.ray_origin_height,
            subsurface_depth=self.subsurface_depth,
        )
        self.ray_rad = self.int_arr.flatten()

        # get an integer index of bands
        self.ray_band_idx = torch.zeros_like(self.int_arr, dtype=torch.int64)
        self.ray_band_idx[:, 10:20] = 1
        self.ray_band_idx[:, 20:80] = 2
        self.ray_band_idx[:, 80:] = 3

        # filter any invalid rays, then normalize and convert the length scale
        self.ray_filter = filter_rays(self.ray_origin, self.ray_dir, self.ray_rad)
        self.ray_origin = self.ray_origin[self.ray_filter]
        self.ray_dir = self.ray_dir[self.ray_filter]
        self.ray_rad = self.ray_rad[self.ray_filter]
        self.ray_len = self.ray_len[self.ray_filter]
        self.ray_alt = self.alt.flatten()[self.ray_filter]
        self.ray_origin_norm, self.scale, self.offset = normalize_rays(
            self.ray_origin, self.ray_dir, self.ray_len
        )
        self.ray_len_norm = self.ray_len / self.scale
        self.ray_band_idx = self.ray_band_idx.flatten()[self.ray_filter]

        # get an integer index into the ray arrays
        self.ray_idx = torch.arange(self.ray_origin_norm.shape[0], dtype=torch.int32)

    def _select_best_slice(self) -> None:
        """Select the best angles to make an RGB image out of this granule."""
        # get the per-band view angles
        sensor_view_angle = torch.from_numpy(
            self.nc_data["sensor_views_bands/sensor_view_angle"][:].data
        ).cuda()
        sensor_view_angle = sensor_view_angle[self.band_order]

        # use the red angles as candidates for the visualization angle
        angles_red = sensor_view_angle[20:80]
        angles_green = sensor_view_angle[10:20]
        angles_blue = sensor_view_angle[:10]

        # get the nearest green and blue angles for each red angle
        idx_green = (
            torch.min((angles_red[:, None] - angles_green[None]) ** 2, dim=1)[1] + 10
        )
        idx_blue = torch.min((angles_red[:, None] - angles_blue[None]) ** 2, dim=1)[1]

        # maximize the minimum number of valid pixels across RGB
        num_valid = (~(self.int_arr.isnan())).sum(dim=0)
        maximizer = (
            torch.stack([num_valid[20:80], num_valid[idx_green], num_valid[idx_blue]])
            .min(dim=0)[0]
            .max(dim=0)[1]
        )
        self.best_rgb_idx = [
            maximizer.item() + 20,
            idx_green[maximizer].item(),
            idx_blue[maximizer].item(),
        ]

    def _parse_ray_data(self) -> None:
        """Parse the relevant data from the netCDF file."""
        self.img_shp = self.nc_data["observation_data/i"].shape[1:]  # image dims

        def _parse_field(
            field: netCDF4.Variable,
        ) -> npt.NDArray[np.float32]:
            """Read a field in HARP2 L1B data, performing the following transformations:
            1) fill invalid values with nan
            2) reshape so the angle dimension is last
            3) flip the y-axis so North is at the top of the image
            4) flatten the image dimensions
            """
            arr = field[:].filled(fill_value=np.nan)
            return np.reshape(np.transpose(arr, (1, 2, 0))[::-1], (-1, 90))

        lat = _parse_field(self.nc_data["geolocation_data/latitude"])
        lon = _parse_field(self.nc_data["geolocation_data/longitude"])
        alt = _parse_field(self.nc_data["geolocation_data/surface_altitude"])
        i = _parse_field(self.nc_data["observation_data/i"])
        # NOTE: don't scale viewing geometry, the scale_factor and add_offset are wrong
        thetav = _parse_field(self.nc_data["geolocation_data/sensor_zenith_angle"])
        phiv = _parse_field(self.nc_data["geolocation_data/sensor_azimuth_angle"])

        # max normalization for intensity
        self.max_i = np.nanmax(i)
        i /= self.max_i

        # reorder the channels, as HARP2 data is in GRNB order but we want BGRN
        self.lat = torch.from_numpy(lat).cuda()[..., self.band_order]
        self.lon = torch.from_numpy(lon).cuda()[..., self.band_order]
        self.alt = torch.from_numpy(alt).cuda()[..., self.band_order]
        self.int_arr = torch.from_numpy(i).cuda()[..., self.band_order]
        self.thetav = torch.from_numpy(thetav).cuda()[..., self.band_order]
        self.phiv = torch.from_numpy(phiv).cuda()[..., self.band_order]

    def get_progress_tracker(self) -> ProgressTracker:
        """Get a ProgressTracker for this HARP2Dataset.

        Returns:
            progress: A ProgressTracker for this HARP2Dataset.
        """
        # get target image by setting valid locations to valid radiances
        target_img = torch.zeros((self.img_shp[0] * self.img_shp[1] * 90)).to(
            self.ray_filter.device
        )
        target_img[self.ray_filter] = self.ray_rad
        target_img = target_img.view(list(self.img_shp) + [90])

        # use the best band/angle indices to minimize striping and cropping
        target_img_rgb = target_img[..., self.best_rgb_idx]

        # set initial predicted image and pixel arrays to zero
        pred_img = torch.zeros_like(target_img)
        pred_pixels = torch.zeros(self.ray_rad.shape)

        progress = ProgressTracker(
            self.ray_filter.view(self.img_shp[0], self.img_shp[1], 90).cpu().numpy(),
            target_img.cpu().numpy(),
            target_img_rgb.cpu().numpy(),
            pred_img.cpu().numpy(),
            pred_pixels.cpu().numpy(),
        )
        return progress

    def get_image_metrics(
        self, pred_img: torch.Tensor, target_img: torch.Tensor
    ) -> dict[str, nn.Module]:
        """Get image quality metrics on HARP2 data.

        Args:
            pred_img: Predicted (reconstructed) image.
            target_img: Target / ground-truth image.

        Returns:
            metrics: A dictionary of metric names and their values.
        """
        data_range = (target_img.max() - target_img.min()).item()
        psnr = torch.zeros(90, device=self.band_order.device)
        ssim = torch.zeros(90, device=self.band_order.device)
        psnr[self.band_order] = peak_signal_noise_ratio(
            pred_img, target_img, dim=(1, 2), reduction="none", data_range=data_range
        )
        _ssim = structural_similarity_index_measure(
            pred_img[:, None], target_img[:, None], reduction="none"
        )
        assert isinstance(_ssim, torch.Tensor)
        ssim[self.band_order] = _ssim

        metrics = {
            "PSNR": psnr.cpu().numpy().tolist(),
            "SSIM": ssim.cpu().numpy().tolist(),
            "PSNR_mean": psnr.mean().item(),
            "SSIM_mean": ssim.mean().item(),
        }
        return metrics

    def get_rgb(self, cube: torch.Tensor) -> torch.Tensor:
        """Get an RGB image from a HARP2 image cube, using the closest to nadir angles.

        Args:
            cube: A HARP2 image cube.

        Returns:
            img: An RGB image of the HARP2 scene.
        """
        assert cube.shape == (90, self.img_shp[0], self.img_shp[1])
        img = torch.clamp(cube[self.best_rgb_idx], 0, 1)
        return img.permute(1, 2, 0).contiguous()

    def get_point_preprocessor(self, point_preprocessor: str) -> Callable:
        """Get a function that preprocesses samples.

        Args:
            point_preprocessor: The type of point preprocessing function to return.
        """
        if point_preprocessor == "horizontal":
            non_nan_lat = self.lat[~self.lat.isnan()]
            non_nan_lon = self.lon[~self.lon.isnan()]

            lat_min, lat_max = non_nan_lat.min(), non_nan_lat.max()
            lon_min, lon_max = non_nan_lon.min(), non_nan_lon.max()
            lat_range, lon_range = lat_max - lat_min, lon_max - lon_min
            alt_range = self.ray_origin_height + self.subsurface_depth

            # if this granule crosses the dateline, shift lon by 180
            shift_lon = lon_max > 179 and lon_min < -179
            if shift_lon:
                non_nan_lon = non_nan_lon % 360 - 180
                lon_min, lon_max = non_nan_lon.min(), non_nan_lon.max()
                lon_range = lon_max - lon_min

            def preprocess_coords(
                coords_xyz: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                dtype = coords_xyz.dtype
                coords_xyz = coords_xyz * self.scale + self.offset
                x, y, z = coords_xyz[..., 0], coords_xyz[..., 1], coords_xyz[..., 2]
                lat, lon, alt = cartesian_to_horizontal(x, y, z)
                subsurface_mask = alt < 0
                if shift_lon:
                    lon = lon % 360 - 180
                lat = 2 * (lat - lat_min) / lat_range - 1
                lon = 2 * (lon - lon_min) / lon_range - 1
                alt = 2 * (alt + self.subsurface_depth) / alt_range - 1
                coords = torch.stack([lat, lon, alt], dim=-1).to(dtype=dtype)
                coords = torch.clip(coords, min=-1, max=1)
                return coords, subsurface_mask

            return preprocess_coords
        else:
            raise NotImplementedError

    def __getitem__(self, idx: int | torch.Tensor) -> dict[str, torch.Tensor]:
        """Get the next item in this HARP2Dataset.

        Args:
            idx: Index of the item to get.

        Returns:
            item: An item in this HARP2Dataset, containing:
                origin: 3D origin point (in normalized scene space) of viewing ray
                dir: 3D direction vector of viewing ray
                alt: Altitude of point where the viewing ray intersects the surface
                rad: Radiance associated with a ray.
                len: Length of viewing ray from origin to surface (in normalized
                    scene space).
                idx: Index of the item in the collection of rays.
        """
        item = {
            "origin": self.ray_origin_norm[idx],
            "dir": self.ray_dir[idx],
            "alt": self.ray_alt[idx],
            "rad": self.ray_rad[idx],
            "len": self.ray_len_norm[idx],
            "idx": self.ray_idx[idx],
            "band_idx": self.ray_band_idx[idx],
        }
        return item

    def __getbatch__(self, idx: torch.Tensor) -> dict[str, torch.Tensor]:
        return self[idx]

    def __len__(self) -> int:
        """Length of the dataset.

        Returns:
            len: Length of this dataset (number of viewing rays).
        """
        len = self.ray_origin_norm.shape[0]
        return len


def download_harp2_file(filename: str, dst_dir: str | Path, level: str) -> None:
    """Use earthaccess to download a HARP2 granule if it is not already present.

    Args:
        filename: Name of the HARP2 granule to download.
        dst_dir: Destination directory where the granule will be downloaded.
        level: Either "L1B" or "L1C", for which level of HARP2 data to download.
    """
    assert level in ["L1B", "L1C"]
    print(f"HARP2 file {filename} not found locally, using Earthaccess to retrieve...")
    earthaccess.login(persist=True)  # prompt user for login
    # get the datetime as a workaround for no direct filename search in earthaccess
    harp2_dt = datetime.strptime(filename.split(".")[1], "%Y%m%dT%H%M%S")
    results = earthaccess.search_data(
        short_name=f"PACE_HARP2_{level}_SCI",
        temporal=(
            harp2_dt.strftime("%Y-%m-%d"),
            (harp2_dt + timedelta(days=1)).strftime("%Y-%m-%d"),
        ),
    )
    if len(results) == 0:
        raise ValueError(f"Could not find {filename} on earthaccess.")
    short_filename = ".".join(filename.split(".")[:4])  # remove file extension
    results = [
        r for r in results if short_filename in r.render_dict["meta"]["native-id"]
    ]
    earthaccess.download(results[0], str(dst_dir))
