"""Extract a voxel grid from a trained neural rendering model."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import warnings

import torch
from torch.cuda import current_device
from tqdm.notebook import tqdm

from atmonr.datasets.factory import BANDS, get_dataset, get_extract_dataset
from atmonr.pipelines.factory import get_pipeline
from atmonr.batch_loader import BatchLoader


if os.getcwd().split("/")[-1] == "notebooks":
    os.chdir("..")


def parse_args() -> argparse.Namespace:
    """Parse, check, and return command-line arguments.

    Returns:
        args: Command-line argument namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-name",
        type=str,
        required=True,
        help="Name of the train.py output directory.",
    )
    parser.add_argument(
        "--coord-mode",
        type=str,
        required=True,
        help="Either 'native' or 'voxelgrid'. The coordinates of the extracted volume "
        "can either match the native resolution / locations of the training data, or "
        "use a user-supplied voxel grid, defined by --horizontal-step.",
    )
    parser.add_argument(
        "--extract-filename",
        type=str,
        required=True,
        help="Name of the output netCDF file containing the extracted volumetric data. "
        "This file will be placed in the experiment directory.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8192, help="Batch size for inference."
    )
    parser.add_argument(
        "--min-alt",
        type=float,
        help="Minimum above-surface altitude (meters) of the voxel grid.",
    )
    parser.add_argument(
        "--max-alt",
        type=float,
        help="Maximum above-surface altitude (meters) of the voxel grid.",
    )
    parser.add_argument(
        "--alt-step",
        type=float,
        default=250.0,
        help="Altitude step size (meters) between adjacent voxels in the same column, "
        "i.e. vertical resolution.",
    )
    parser.add_argument(
        "---horizontal-step",
        type=float,
        default=3000.0,
        help="Horizontal step size (meters) between adjacent voxels at the same "
        "altitude, i.e. horizontal resolution.",
    )
    parser.add_argument
    args = parser.parse_args()
    assert Path(f"data/output/{args.exp_name}").exists()
    assert args.coord_mode in ["native", "voxelgrid"]
    assert args.alt_step > 0 and args.horizontal_step > 0
    if args.alt_step <= 50:
        warnings.warn(
            f"Provided --alt-step of {args.alt_step} is very low and may "
            "cause this script to run for a long time."
        )
    if args.horizontal_step <= 500:
        warnings.warn(
            f"Provided --horizontal-step of {args.horizontal_step} is very "
            "low and may cause this script to run for a long time."
        )
    return args


def main() -> None:
    args = parse_args()

    output_path = Path(f"data/output/{args.exp_name}")

    # load training arguments and config
    train_args = SimpleNamespace(**json.load(open(output_path / "args.json")))
    config = json.load(open(output_path / "config.json"))
    device = current_device()  # only support single-gpu training for now

    if not args.min_alt:
        args.min_alt = -config["pipeline"]["subsurface_depth"]
    if not args.max_alt:
        args.max_alt = config["pipeline"]["ray_origin_height"]

    # get the dataset
    dataset = get_dataset(
        config["data_type"],
        train_args.scene_filename,
        ray_origin_height=config["pipeline"]["ray_origin_height"],
        subsurface_depth=config["pipeline"]["subsurface_depth"],
    )

    # get the sample altitudes
    sample_alt = torch.arange(
        args.min_alt, args.max_alt + args.alt_step / 2, args.alt_step
    ).to(device)

    extract_dataset = get_extract_dataset(
        args.coord_mode,
        config["data_type"],
        dataset,
        args.horizontal_step,
        sample_alt,
    )
    dataloader = BatchLoader(
        extract_dataset, batch_size=args.batch_size * sample_alt.shape[0], shuffle=False
    )

    # set up the neural rendering pipeline and set it to evaluation mode
    pipeline = get_pipeline(config["pipeline"], dataset)
    pipeline.send_tensors_to(device)
    pipeline.eval()

    # get most recent checkpoint using epoch number and load it
    ckpts = list(output_path.glob("epoch_*.pt"))
    last_ckpt_path = sorted(ckpts, key=lambda c: int(c.stem.split("_")[1]))[-1]
    pipeline.load_state_dict(torch.load(last_ckpt_path)["pipeline"])

    sigma = torch.zeros(
        (extract_dataset.idx.shape[0], BANDS[config["data_type"]]), device=device
    )

    for batch in tqdm(dataloader):
        xyz = batch["xyz"]
        pts = (xyz - dataset.offset) / dataset.scale

        with torch.no_grad():
            sigma_batch = pipeline.extract(pts).to(dtype=sigma.dtype)
            sigma[batch["idx"]] = sigma_batch.detach() / dataset.scale

    extract_dataset.write_netcdf(output_path / args.extract_filename, sigma)


if __name__ == "__main__":
    main()
