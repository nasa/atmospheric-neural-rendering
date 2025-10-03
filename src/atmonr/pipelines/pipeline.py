from typing import Any, Mapping
import warnings

import torch
from torch.optim import Optimizer

from atmonr.datasets.factory import Dataset


class Pipeline:
    """Base class for Pipelines, which process batches of rays into output densities and
    colors. Pipeline is responsible for maintaining the internal state of its neural
    rendering algorithm, as well as for defining the loss function and optimizer. To
    implement a pipeline, inherit this class and make sure all of the stub methods below
    are implemented.
    """

    def __init__(
        self,
        config: dict,
        dataset: Dataset,
    ) -> None:
        """Common init behavior across all pipelines. This should get called at the top
        of any inheriting class __init__ method.

        Args:
            config: Configuration options for this pipeline.
            dataset: Dataset to which this pipeline will be applied.
        """
        self.ray_origin_height = dataset.config["ray_origin_height"]
        assert not (
            config["point_preprocessor"] == "horizontal" and config["include_height"]
        )
        if (
            not config["point_preprocessor"]
            and isinstance(config["encoder"]["L_x"], list)
            and not all(
                [
                    num_freqs == config["encoder"]["L_x"][0]
                    for num_freqs in config["encoder"]["L_x"]
                ]
            )
        ):
            warnings.warn(
                "Are you sure you want to use a variable encoding dimension for "
                "non-transformed coordinates?"
            )

        self.device = -1
        self.config = config

        self.scale = dataset.scale
        self.offset = dataset.offset

        if self.config["point_preprocessor"]:
            self.point_preprocessor = dataset.get_point_preprocessor(
                self.config["point_preprocessor"]
            )
        else:
            self.point_preprocessor = None

    def send_tensors_to(self, device: int) -> None:
        raise NotImplementedError

    def get_optimizer(
        self,
        config: dict,
    ) -> Optimizer:
        raise NotImplementedError

    def forward(self, ray_batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def extract(self, pts: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def compute_loss(
        self, ray_batch: Mapping[str, torch.Tensor], results: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        raise NotImplementedError

    def state_dict(self) -> Mapping[str, Mapping[str, Any]]:
        raise NotImplementedError

    def load_state_dict(self, state_dict: dict) -> None:
        raise NotImplementedError

    def train(self) -> None:
        raise NotImplementedError

    def eval(self) -> None:
        raise NotImplementedError
