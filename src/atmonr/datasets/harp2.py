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

from atmonr.geospatial.wgs_84 import (
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
        config: dict,
        filename: str,
        chunk_size: int = int(1e4),
    ) -> None:
        """Initialize a HARP2Dataset.

        Args:
            config: Config options for this dataset.
            filename: Filename corresponding to a HARP2 file.
            chunk_size: Number of pixels to process at once. Use this to cut down on GPU
                memory overhead during startup.
        """
        super().__init__()
        self.config = config
        self.filename = filename
        self.local_path = Path("data/HARP2") / filename

        if "max_abs_view_angle" not in self.config:
            self.config["max_abs_view_angle"] = 90.0

        # if the file does not exist, download it
        if not self.local_path.exists():
            download(self.filename, self.local_path.parent, "L1B")

        # load the netCDF4 file
        self.nc_data = netCDF4.Dataset(self.local_path)

        # get the index into the views to resort into IRGB order
        self.view_idx, self.irgb_idx = get_indexes(
            self.nc_data,
            self.config["max_abs_view_angle"],
            self.config["bands_to_keep"],
        )

        self._init_data()
        self._init_rgb_idxs()
        self._init_ray_data(chunk_size)

    def _init_data(self) -> None:
        """Parse the relevant data from the netCDF file."""
        level = self.nc_data.processing_level
        assert level in ["L1B", "L1C"]
        if level == "L1B":
            self.img_shp = self.nc_data["observation_data/i"].shape[1:]
        else:
            self.img_shp = self.nc_data["observation_data/i"].shape[:2]

        def _parse_field(
            field: netCDF4.Variable,
        ) -> npt.NDArray[np.float32]:
            """Read a field in HARP2 L1B/L1C data, making sure that:
            1) invalid values are filled with nan
            2) views are filtered and in IRGB order
            3) North is at the top of the image
            4) the angle dimension is last
            5) the image dimensions are flattened
            """
            arr = field[:].filled(fill_value=np.nan)
            assert len(arr.shape) >= 2 and len(arr.shape) <= 4
            nv = self.view_idx.shape[0]  # number of views kept
            if level == "L1B":
                return arr[self.view_idx, ::-1].transpose((1, 2, 0)).reshape((-1, nv))
            if len(arr.shape) == 4:
                arr = arr[..., 0]
            if len(arr.shape) == 3:
                return arr[::-1, :, self.view_idx].reshape((-1, nv))
            return np.tile(arr[::-1, :, None], (1, 1, self.view_idx.shape[0])).reshape(
                (-1, nv)
            )

        lat = _parse_field(self.nc_data["geolocation_data/latitude"])
        lon = _parse_field(self.nc_data["geolocation_data/longitude"])
        if level == "L1B":
            alt = _parse_field(self.nc_data["geolocation_data/surface_altitude"])
        else:
            alt = _parse_field(self.nc_data["geolocation_data/height"])
        thetav = _parse_field(self.nc_data["geolocation_data/sensor_zenith_angle"])
        phiv = _parse_field(self.nc_data["geolocation_data/sensor_azimuth_angle"])
        i = _parse_field(self.nc_data["observation_data/i"])

        # maximum intensity value required for visualization purposes later
        self.max_i = np.nanmax(i).item()

        # move everything to device
        self.lat = torch.from_numpy(lat).cuda()
        self.lon = torch.from_numpy(lon).cuda()
        self.alt = torch.from_numpy(alt).cuda()
        self.int_arr = torch.from_numpy(i).cuda()
        self.thetav = torch.from_numpy(thetav).cuda()
        self.phiv = torch.from_numpy(phiv).cuda()

    def _init_rgb_idxs(self, mode: str = "nadir") -> None:
        """Select the best angles to make an RGB image out of this granule.
        
        Args:
            mode: TODO: describe and make configurable
        """
        angles = self.nc_data["sensor_views_bands/sensor_view_angle"][
            self.view_idx
        ].filled(fill_value=np.nan)
        num_valid = (~(self.int_arr.isnan())).sum(dim=0).cpu().numpy()
        striped = np.zeros_like(num_valid, dtype=bool)
        if self.nc_data.processing_level == "L1B":
            # simple way to check for striped views in L1B
            striped = num_valid < num_valid.mean()
        masks_rgb = [self.irgb_idx == i for i in range(1, 4)]
        idxs_rgb = [np.where(mask)[0] for mask in masks_rgb]
        angles_rgb = [angles[mask] for mask in masks_rgb]

        # if red is missing, use closest-to-nadir non-striped
        if not masks_rgb[0].any():
            best_idx = np.argmin(np.abs(angles) + striped * 1000).item()
            self.best_rgb_idx = [best_idx, best_idx, best_idx]
            return
        # if green and/or blue is missing but red is present, use grayscale red
        if not masks_rgb[1].any() or not masks_rgb[2].any():
            best_idx = idxs_rgb[0][np.argmin(np.abs(angles_rgb[0]) + striped[masks_rgb[0]] * 1000).item()].item()
            self.best_rgb_idx = [best_idx, best_idx, best_idx]
            return

        # find the green/blue view angles which minimize the RGB aberration
        angles_rgb_mg = np.stack(np.meshgrid(*angles_rgb, indexing="ij"))
        angle_ranges = angles_rgb_mg.max(axis=0) - angles_rgb_mg.min(axis=0)
        idx_nearest = angle_ranges.reshape((angles_rgb[0].shape[0], -1)).argmin(axis=1)
        idx_nearest_green = idxs_rgb[1][idx_nearest // angles_rgb[2].shape[0]]
        idx_nearest_blue = idxs_rgb[2][idx_nearest % angles_rgb[2].shape[0]]

        if mode == "nadir":
            # get the closest to nadir, while dodging striped angles
            nadir_idx_red = np.argmin(np.abs(angles_rgb[0]) + striped[masks_rgb[0]] * 1000).item()
            # the green and blue _shouldn't_ be striped, though this doesn't check
            self.best_rgb_idx = [
                idxs_rgb[0][nadir_idx_red].item(),
                idx_nearest_green[nadir_idx_red].item(),
                idx_nearest_blue[nadir_idx_red].item(),
            ]
        elif mode == "most_pixels":        
            # maximize across indices the minimum number across RGB of valid pixels
            maximizer = (
                np.stack(
                    [
                        num_valid[masks_rgb[0]],
                        num_valid[idx_nearest_green],
                        num_valid[idx_nearest_blue],
                    ]
                )
                .min(axis=0)
                .argmax(axis=0)
                .item()
            )
            self.best_rgb_idx = [
                idxs_rgb[0][maximizer].item(),
                idx_nearest_green[maximizer].item(),
                idx_nearest_blue[maximizer].item(),
            ]
        else:
            raise NotImplementedError(f"Unrecognized RGB indexing mode {mode}")

    def _init_ray_data(self, chunk_size: int) -> None:
        """Initialize the ray data for this dataset. Chunking is used to lower the size
        of the cache pytorch creates, easing the GPU memory load.
        
        Args:
            chunk_size: Number of rays per chunk.
        """
        # convert from image-like data to a flattened array of rays
        num_rays = self.lat.shape[0] * self.lat.shape[1]
        self.ray_origin = torch.zeros(
            (num_rays, 3), dtype=torch.float32, device=self.lat.device
        )
        self.ray_dir = torch.zeros(
            (num_rays, 3), dtype=torch.float32, device=self.lat.device
        )
        self.ray_len = torch.zeros(
            (num_rays,), dtype=torch.float32, device=self.lat.device
        )

        # call get_rays in chunks to minimize memory overhead of rotation matrix tensor
        total_rays = 0
        for chunk_idx in range(-(-self.lat.shape[0] // chunk_size)):
            slc_in = slice(
                chunk_idx * chunk_size,
                min((chunk_idx + 1) * chunk_size, self.lat.shape[0]),
            )
            chunk_origin, chunk_dir, chunk_len = get_rays(
                self.lat[slc_in],
                self.lon[slc_in],
                self.alt[slc_in],
                self.thetav[slc_in],
                self.phiv[slc_in],
                ray_origin_height=self.config["ray_origin_height"],
            )
            num_chunk_rays = chunk_origin.shape[0]
            slc_out = slice(total_rays, total_rays + num_chunk_rays)
            self.ray_origin[slc_out] = chunk_origin
            self.ray_dir[slc_out] = chunk_dir
            self.ray_len[slc_out] = chunk_len
            total_rays += num_chunk_rays
        self.ray_rad = self.int_arr.flatten()

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
        self.ray_irgb_idx = torch.from_numpy(self.irgb_idx).to(
            device=self.ray_filter.device
        )[torch.where(self.ray_filter.view((-1, self.view_idx.shape[0])))[1]]
        # get an integer index into the ray arrays
        self.ray_idx = torch.arange(self.ray_origin_norm.shape[0], dtype=torch.int32)

    def get_progress_tracker(self) -> ProgressTracker:
        """Get a ProgressTracker for this HARP2Dataset.

        Returns:
            progress: A ProgressTracker for this HARP2Dataset.
        """
        # get target image by setting valid locations to valid radiances
        target_img = torch.zeros(
            (self.img_shp[0] * self.img_shp[1] * self.view_idx.shape[0])
        ).to(self.ray_filter.device)
        target_img[self.ray_filter] = self.ray_rad
        target_img = target_img.view(list(self.img_shp) + [self.view_idx.shape[0]])
        target_img_rgb = self.get_rgb(target_img.permute((2, 0, 1)))

        # set initial predicted image and pixel arrays to zero
        pred_img = torch.zeros_like(target_img)
        pred_pixels = torch.zeros(self.ray_rad.shape)
        pred_img_surf = torch.zeros_like(target_img)
        pred_pixels_surf = torch.zeros(self.ray_rad.shape)
        pred_img_atmo = torch.zeros_like(target_img)
        pred_pixels_atmo = torch.zeros(self.ray_rad.shape)

        progress = ProgressTracker(
            self.ray_filter.view(
                self.img_shp[0], self.img_shp[1], self.view_idx.shape[0]
            ).cpu().numpy(),
            target_img.cpu().numpy(),
            target_img_rgb.cpu().numpy(),
            pred_img.cpu().numpy(),
            pred_pixels.cpu().numpy(),
            pred_img_surf.cpu().numpy(),
            pred_pixels_surf.cpu().numpy(),
            pred_img_atmo.cpu().numpy(),
            pred_pixels_atmo.cpu().numpy(),
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
        pred_img, target_img = pred_img / self.max_i, target_img / self.max_i

        # clip image before image metrics since intensities are max-normalized
        pred_img = torch.clip(pred_img, min=0, max=1)

        data_range = (target_img.max() - target_img.min()).item()
        psnr = (
            torch.zeros(self.view_idx.shape[0], device=self.int_arr.device) + torch.nan
        )
        ssim = (
            torch.zeros(self.view_idx.shape[0], device=self.int_arr.device) + torch.nan
        )
        psnr = peak_signal_noise_ratio(
            pred_img, target_img, dim=(1, 2), reduction="none", data_range=data_range
        )
        ssim = structural_similarity_index_measure(
            pred_img[:, None], target_img[:, None], reduction="none"
        )
        assert isinstance(ssim, torch.Tensor)

        metrics = {
            "PSNR": psnr.cpu().numpy().tolist(),
            "SSIM": ssim.cpu().numpy().tolist(),
            "PSNR_mean": psnr[~torch.isnan(psnr)].mean().item(),
            "SSIM_mean": ssim[~torch.isnan(ssim)].mean().item(),
        }
        return metrics

    def get_rgb(self, cube: torch.Tensor) -> torch.Tensor:
        """Get an RGB image from a HARP2 image cube, using the best RGB index.

        Args:
            cube: A HARP2 image cube.

        Returns:
            img: An RGB image of the HARP2 scene.
        """
        assert cube.shape == (self.view_idx.shape[0], self.img_shp[0], self.img_shp[1])
        img = torch.clamp(cube[self.best_rgb_idx] / self.max_i, 0, 1)
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

            # if this granule crosses the dateline, shift lon by 180
            shift_lon = lon_max > 179 and lon_min < -179
            if shift_lon:
                non_nan_lon = non_nan_lon % 360 - 180
                lon_min, lon_max = non_nan_lon.min(), non_nan_lon.max()
                lon_range = lon_max - lon_min

            def preprocess_coords(
                coords_xyz: torch.Tensor,
            ) -> torch.Tensor:
                dtype = coords_xyz.dtype
                coords_xyz = coords_xyz * self.scale + self.offset
                x, y, z = coords_xyz[..., 0], coords_xyz[..., 1], coords_xyz[..., 2]
                lat, lon, alt = cartesian_to_horizontal(x, y, z)
                if shift_lon:
                    lon = lon % 360 - 180
                lat = 2 * (lat - lat_min) / lat_range - 1
                lon = 2 * (lon - lon_min) / lon_range - 1
                alt = 2 * alt / self.config["ray_origin_height"] - 1
                coords = torch.stack([lat, lon, alt], dim=-1).to(dtype=dtype)
                coords = torch.clip(coords, min=-1, max=1)
                return coords

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
            "irgb_idx": self.ray_irgb_idx[idx],
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


def download(filename: str, dst_dir: str | Path, level: str) -> None:
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


def get_indexes(
    nc_data: netCDF4.Dataset,
    max_abs_view_angle: float,
    bands_to_keep: list = [0, 1, 2, 3],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Get an index of which view angles are within a maximum absolute value threshold,
    as well as the index from each view into the band index.
    
    Args:
        nc_data: The netCDF data of a HARP2 L1B or L1C file.
        max_abs_view_angle: The maximum absolute viewing angle to allow.
        bands_to_keep: The list of bands to keep, with 0: infrared, 1: red, 2: green,
            3: blue. Defaults to [0, 1, 2, 3].

    Returns:
        view_idx: Index of views meeting the threshold in the original netCDF order.
        irgb_idx: Index of the band for each view, with 0: infrared, 1: red, 2: green,
            3: blue.
    """
    if nc_data.processing_level not in ["L1B", "L1C"]:
        raise NotImplementedError(
            "Not implemented for level {nc_data.processing_level} data!"
        )
    # get a mask of which views to use
    angles = nc_data["sensor_views_bands/sensor_view_angle"][:].filled(
        fill_value=np.nan
    )
    angles_filtered = np.where(np.abs(angles) <= max_abs_view_angle)[0]
    # get the index from all 90 into the IRGB-sorted arrays
    wavelengths = nc_data["sensor_views_bands/intensity_wavelength"][:].data.flatten()
    view_order = np.argsort(-wavelengths, stable=True)  # sort by decreasing wavelength
    view_idx = view_order[np.isin(view_order, angles_filtered)]
    irgb_idx = np.where(
        wavelengths[view_idx, None] == np.unique(wavelengths)[None, ::-1]
    )[1]

    mask_bands_to_keep = np.isin(irgb_idx, bands_to_keep)

    view_idx = view_idx[mask_bands_to_keep]
    irgb_idx = irgb_idx[mask_bands_to_keep]
    return view_idx, irgb_idx
