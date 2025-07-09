"""Train a neural rendering model to fit to multi-angle satellite data."""

import argparse
import json
import os
from pathlib import Path

from torch.cuda import current_device

from atmonr.datasets.factory import get_dataset
from atmonr.pipelines.factory import get_pipeline
from atmonr.trainer import Trainer
from atmonr.utils import load_config


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
        help="Name of this experiment.",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to the configuration for this experiment.",
    )
    parser.add_argument(
        "--scene-filename",
        type=str,
        required=True,
        help="Filename of the scene to reconstruct.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Use the pytorch profiler to analyze code performance.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite experiment directory if it exists.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted experiment on the next epoch.",
    )
    args = parser.parse_args()
    return args


def setup_dir(args: argparse.Namespace, config: dict) -> Path:
    """Set up the directory for this training run.

    Args:
        args: Script-level arguments.
        config: Config dictionary.

    Returns:
        output_path: Path to the output directory
    """
    output_path = Path(f"data/output/{args.exp_name}")
    if args.resume:
        assert output_path.exists()
    else:
        assert args.overwrite or not output_path.exists()
    os.makedirs(output_path, exist_ok=True)
    json.dump(vars(args), open(f"{output_path}/args.json", "w"), indent=4)
    json.dump(config, open(f"{output_path}/config.json", "w"), indent=4)
    return output_path


def main() -> None:
    args = parse_args()
    config = load_config(args.config_path)
    output_path = setup_dir(args, config)

    device = current_device()  # only support single-gpu training for now

    # get the dataset
    dataset = get_dataset(
        config["data_type"],
        args.scene_filename,
        ray_origin_height=config["pipeline"]["ray_origin_height"],
        subsurface_depth=config["pipeline"]["subsurface_depth"],
    )

    # set up the neural rendering pipeline and the trainer
    pipeline = get_pipeline(config["pipeline"], dataset)
    pipeline.send_tensors_to(device)
    trainer = Trainer(config["trainer"], dataset, pipeline, args.exp_name)

    # allow resuming interrupted experiments
    if args.resume:
        trainer.load(output_path)

    trainer.train(output_path, profile=args.profile)


if __name__ == "__main__":
    main()
