"""Miscellaneous utilities."""

from typing import Mapping
import json

import torch
from torch._prims_common import DeviceLikeType


def load_config(config_path: str) -> dict:
    """Load a config file and fix some common mistakes.

    Args:
        config_path: Path to the config file for this training run.
    """
    config = json.load(open(config_path))
    if config["pipeline"]["type"].lower() == "nerf":
        config["pipeline"]["type"] = "NeRF"
    if config["data_type"].lower() == "harp2":
        config["data_type"] = "HARP2"
    return config


def dict_to(
    d: dict[str, torch.Tensor], device: DeviceLikeType
) -> Mapping[str, torch.Tensor]:
    """Send a dictionary's values to the provided device.

    Args:
        d: The dictionary to convert.
        device: The destination device.

    Returns:
        d: The converted dictionary.
    """
    for k in d:
        d[k] = d[k].to(device)
    return d
