"""Make a video of the learned volume density / extinction coefficient in a scene.

Notices:
The copyright notice below, to be included in the software, has also been provided in the license. 
 
“Copyright © 2023 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.”
 
No other release is authorized at this time.
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import warnings

import netCDF4
import numpy as np

try:
    import pyopenvdb as vdb  # type: ignore
except ImportError:
    try:
        import openvdb as vdb  # type: ignore
    except ImportError:
        raise ImportError(
            "You must have openvdb Python bindings installed to use make_video.py"
        )
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    """Parse, check, and return command-line arguments.

    Returns:
        args: Command-line argument namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--extract-filepath",
        type=str,
        required=True,
        help="Path of a netCDF file with an extracted volume.",
    )
    parser.add_argument(
        "--vdb-filepath",
        type=str,
        required=True,
        help="Path where the VDB data will be saved.",
    )
    parser.add_argument(
        "--video-filepath",
        type=str,
        required=True,
        help="Path where the rendered video will be saved.",
    )
    parser.add_argument(
        "--render-band-idx", type=int, help="Index of the band to render."
    )
    parser.add_argument(
        "--res",
        type=str,
        default="640x480",
        help="Resolution at which to render the video.",
    )
    parser.add_argument(
        "--frame-rate",
        type=int,
        default=60,
        help="Frame rate at which to render the video.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Duration in seconds of the video.",
    )
    parser.add_argument(
        "--absorb",
        nargs=3,
        type=float,
        default=(
            0.1,
            0.1,
            0.1,
        ),
        help="Absorption coefficients for vdb_render.",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=0.01,
        help="Density and transmittance cutoff value for vdb_render.",
    )
    parser.add_argument(
        "--light-source-dir",
        nargs=3,
        type=float,
        default=(
            0.0,
            1.0,
            0.0,
        ),
        help="Light source direction for vdb_render.",
    )
    parser.add_argument(
        "--light-source-color",
        nargs=3,
        type=float,
        default=(
            1.0,
            1.0,
            1.0,
        ),
        help="Light color for vdb_render.",
    )
    parser.add_argument(
        "--scatter",
        nargs=3,
        type=float,
        default=(
            0.7,
            0.7,
            0.7,
        ),
        help="Scattering coefficients for vdb_render.",
    )
    args = parser.parse_args()
    assert Path(args.extract_filepath).exists()
    args.res = [int(pix) for pix in args.res.split("x")]
    assert len(args.res) == 2 and args.res[0] > 0 and args.res[1] > 0
    if args.res[0] * args.res[1] > 1920 * 1080:
        warnings.warn(
            f"Attempting to render with a large resolution of {args.res} will be very "
            "slow and may cause vdb_render to run out of memory."
        )
    assert args.duration > 0
    return args


def main():
    args = parse_args()
    ncdata = netCDF4.Dataset(args.extract_filepath)
    # flip the netCDF data along the altitude dimension
    sigma = ncdata["extinction_coefficient"][:, :, ::-1, 2].filled(fill_value=np.nan)
    # swap axes so we're not in left-handed coords and so height is on the y-axis
    sigma = np.ascontiguousarray(np.transpose(sigma, (1, 2, 0)))

    grid = vdb.FloatGrid()
    # km scale works well with vdb_render
    grid.copyFromArray(sigma * ncdata.neural_rendering_scene_scale / 1000)
    vdb.write(args.vdb_filepath, grids=[grid])

    num_frames = int(args.duration * args.frame_rate)
    times = np.linspace(0, args.duration, num_frames)
    center = (sigma.shape[0] / 2, sigma.shape[1] / 2, sigma.shape[2] / 2)

    # get orbit locations
    orbit_radius = 1.3 * np.linalg.norm(sigma.shape)
    t_circle = 2 * np.pi * times / args.duration
    orbit_x = np.cos(t_circle) * orbit_radius + center[0]
    orbit_y = np.sin(t_circle) * orbit_radius + center[2]

    # height at which to fix camera
    view_height = 0.5 * np.linalg.norm(sigma.shape)

    # look at the bottom center of the scene
    lookat = f"{center[0]},0,{center[2]}"

    if Path("_temp_frames").exists():
        shutil.rmtree("_temp_frames")
    os.makedirs("_temp_frames", exist_ok=True)
    for idx in tqdm(range(times.shape[0]), total=times.shape[0]):
        frame_file = f"_temp_frames/{idx:06d}.ppm"
        res = "x".join([str(pix) for pix in args.res])
        absorb = ",".join([str(el) for el in args.absorb])
        light = ",".join(
            [
                str(el)
                for el in list(args.light_source_dir) + list(args.light_source_color)
            ]
        )
        scatter = ",".join([str(el) for el in args.scatter])
        vdb_render_command = (
            f"vdb_render {args.vdb_filepath} {frame_file} -compression none -lookat {lookat} "
            f"-translate {orbit_x[idx]},{view_height},{orbit_y[idx]} -res {res} -absorb"
            f" {absorb} -cutoff {args.cutoff} -light {light} -scatter {scatter}"
        )
        subprocess.run(
            vdb_render_command.split(" "),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
    ffmpeg_command = f"ffmpeg -framerate 60 -i '_temp_frames/%06d.ppm' -c:v libx264 -pix_fmt yuv420p -s 640x480 -y {args.video_filepath}"
    subprocess.run(
        ffmpeg_command.split(" "), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
    )
    shutil.rmtree("_temp_frames")


if __name__ == "__main__":
    main()
