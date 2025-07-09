from itertools import chain
from typing import Any, Mapping

import tinycudann as tcnn
import torch
import torch.nn.functional as F
from torch.optim import AdamW, Optimizer
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn  # type: ignore

from atmonr.datasets.factory import Dataset
from atmonr.pipelines.pipeline import Pipeline
from atmonr.render import render
from atmonr.samplers import append_heights, sample_uniform_bins


class InstantNGPPipeline(Pipeline):
    """This pipeline uses an approach based on Instant Neural Graphics Primitives
    (Instant-NGP) to render an atmospheric scene. This implementation differs in several
    ways, mostly accounting for the differences between pinhole camera imagery and
    atmospheric satellite data. This approach does not use an occupancy grid.

    Instant-NGP paper:
    https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf
    """

    def __init__(
        self,
        config: dict,
        dataset: Dataset,
    ) -> None:
        """Initialize an InstantNGPPipeline.

        Args:
            config: Configuration options for this InstantNGPPipeline.
            dataset: Dataset to which this pipeline will be applied.
        """
        super().__init__(config, dataset)

        num_inputs = 4 if self.config["include_height"] else 3

        pos_encoder = tcnn.Encoding(
            num_inputs,
            self.config["instant_ngp"]["encoding"],
        )
        pos_mlp = tcnn.Network(
            pos_encoder.n_output_dims,
            16,
            self.config["instant_ngp"]["network"],
        )
        self.pos_model = torch.nn.Sequential(pos_encoder, pos_mlp)

        dir_encoder = tcnn.Encoding(
            3 + 16,
            self.config["instant_ngp"]["dir_encoding"],
        )
        dir_mlp = tcnn.Network(
            dir_encoder.n_output_dims,
            self.config["num_bands"],
            self.config["instant_ngp"]["rgb_network"],
        )
        self.dir_model = torch.nn.Sequential(dir_encoder, dir_mlp)

        if self.config["instant_ngp"]["ema_decay"]:
            ema_multi_avg_fn = get_ema_multi_avg_fn(
                self.config["instant_ngp"]["ema_decay"]
            )
            self.avg_pos_model = AveragedModel(
                self.pos_model,
                multi_avg_fn=ema_multi_avg_fn,  # type: ignore
            )
            self.avg_dir_model = AveragedModel(
                self.dir_model,
                multi_avg_fn=ema_multi_avg_fn,  # type: ignore
            )
        else:
            self.avg_pos_model = None
            self.avg_dir_model = None
        self.training = True

    def send_tensors_to(self, device: int) -> None:
        """Move the relevant tensors to a CUDA-capable device.

        Args:
            device: The index of a CUDA-capable device.
        """
        self.device = device

    def get_optimizer(self, config: dict) -> Optimizer:
        """Get the optimizer for this InstantNGPPipeline, which is Adam with weight
        decay for only the MLP parameters, not the hash table."""
        enc_params = chain(
            self.pos_model[0].parameters(), self.dir_model[0].parameters()
        )
        mlp_params = chain(
            self.pos_model[1].parameters(), self.dir_model[1].parameters()
        )
        optimizer = AdamW(
            [
                {"params": enc_params, "weight_decay": 0},
                {"params": mlp_params},
            ],
            **config,
        )
        return optimizer

    def forward(self, ray_batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run Instant-NGP on the provided batch.
        Args:
            ray_batch: A batch of rays and associated observations.

        Returns:
            results: Results, including densities and colors.
        """
        B_ = ray_batch["origin"].shape[0]
        N = 192
        pts, z_vals = sample_uniform_bins(ray_batch, n_bins=N)

        if self.point_preprocessor:
            pts = self.point_preprocessor(pts)

        # Instant-NGP uses [0, 1], not [-1, 1]
        pts = (pts + 1) / 2

        # add height above surface to the points vector, if specified in the config
        if self.config["include_height"]:
            pts = append_heights(
                pts, self.config["ray_origin_height"], self.scale, self.offset
            )

        # repeat the direction vector N times to match the points vector
        dirs = ray_batch["dir"][:, None].repeat(1, N, 1).view(B_ * N, 3)

        # apply the MLPs, reshape color output
        pos_out = self.pos_model(pts.view(B_ * N, -1))
        color = self.dir_model(torch.cat([dirs, pos_out], dim=1))
        color = color.view(B_, N, self.config["num_bands"])

        # the first num_bands values of the intermediate output are treated as densities
        sigma = pos_out[..., : self.config["num_bands"]].view(B_, N, -1)

        # volume rendering
        color_map, weights = render(z_vals, color, sigma)

        results = {
            "color_fine": color,
            "sigma_fine": sigma,
            "color_map_fine": color_map,
            "weights_fine": weights,
            "z_vals_fine": z_vals,
        }
        if self.config["include_height"]:
            results["norm_heights_fine"] = pts[..., 3]
        return results

    def extract(self, pts: torch.Tensor) -> torch.Tensor:
        """Extract the extinction coefficient for the provided points.
        Args:
            pts: Array of 3D points in normalized scene frame, of shape (batch_size *
                num_samples, 3).
            scale: The scale in meters of the scene.
            offset: The offset (x,y,z) in meters of the scene.

        Returns:
            sigma: Extinction coefficient at the provided points.
        """
        # if we have a point preprocessing function, use it
        if self.point_preprocessor:
            pts = self.point_preprocessor(pts[None])[0]
        # Instant-NGP uses [0, 1], not [-1, 1]
        pts = (pts + 1) / 2
        # add height above surface to the points vector, if specified in the config
        if self.config["include_height"]:
            pts = append_heights(
                pts[None], self.config["ray_origin_height"], self.scale, self.offset
            )[0]

        pos_out = self.pos_model(pts)

        # the first num_bands values of the intermediate output are treated as densities
        sigma = torch.clip(
            pos_out[..., : self.config["num_bands"]].view(
                pts.shape[0], self.config["num_bands"]
            ),
            min=0,
        )
        return sigma

    def compute_loss(
        self, ray_batch: Mapping[str, torch.Tensor], results: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute the Huber loss.
        Args:
            ray_batch: A batch of rays and associated observations.
            results: Results from the forward pass of Instant NGP.
        Returns:
            loss: Huber loss of results with respect to this batch.
        """
        results_indexed = torch.take_along_dim(
            results["color_map_fine"], ray_batch["band_idx"][:, None], 1
        )[:, 0]
        loss = F.huber_loss(results_indexed, ray_batch["rad"])
        return loss

    def update_parameters(self) -> None:
        """Update parameters with weight averaging."""
        if self.avg_pos_model:
            self.avg_pos_model.update_parameters(self.pos_model)
        if self.avg_dir_model:
            self.avg_dir_model.update_parameters(self.dir_model)

    def state_dict(self) -> Mapping[str, Mapping[str, Any]]:
        """Get the state dictionary of this InstantNGPPipeline.

        Returns:
            state_dict: Nested dict with all state_dict's in this InstantNGPPipeline.
        """
        state_dict = {}
        if self.avg_pos_model:
            state_dict["avg_pos_model"] = self.avg_pos_model.state_dict()
        else:
            state_dict["pos_model"] = self.pos_model.state_dict()
        if self.avg_dir_model:
            state_dict["avg_dir_model"] = self.avg_dir_model.state_dict()
        else:
            state_dict["dir_model"] = self.dir_model.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the weights from a state dictionary into this InstantNGPPipeline.

        Args:
            state_dict: State dict of a InstantNGPPipeline.
        """
        if self.avg_pos_model:
            self.avg_pos_model.load_state_dict(state_dict["avg_pos_model"])
            self.pos_model = self.avg_pos_model.module[0]
        else:
            self.pos_model.load_state_dict(state_dict["pos_model"])
        if self.avg_dir_model:
            self.avg_dir_model.load_state_dict(state_dict["avg_dir_model"])
            self.dir_model = self.avg_dir_model.module[0]
        else:
            self.dir_model.load_state_dict(state_dict["dir_model"])

    def train(self) -> None:
        """Set this InstantNGPPipeline to training mode."""
        self.training = True
        self.pos_model.train()
        self.dir_model.train()

    def eval(self) -> None:
        """Set this InstantNGPPipeline to evaluation mode."""
        self.training = False
        self.pos_model.eval()
        self.dir_model.eval()
