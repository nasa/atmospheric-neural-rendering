from itertools import chain
from typing import Any, Mapping

import tinycudann as tcnn
import torch
import torch.nn.functional as F
from torch.optim import AdamW, Optimizer

from atmonr.datasets.factory import Dataset
from atmonr.graphics_utils import render_with_surface
from atmonr.losses import (
    dark_loss,
    hdr_loss,
    l1_loss,
    l1_plus_hdr_loss,
    mse_loss,
    mse_plus_hdr_loss,
)
from atmonr.pipelines.pipeline import Pipeline
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

        self.num_density_outputs = 1
        if self.config["multi_band_extinction"]:
            self.num_density_outputs = self.config["num_bands"]

        num_inputs = 4 if self.config["include_height"] else 3

        self.module_names = [
            "pos_encoder",
            "pos_mlp",
            "dir_encoder",
            "dir_mlp",
            "surf_encoder",
            "surf_mlp",
        ]
        self.pos_encoder = tcnn.Encoding(
            num_inputs,
            self.config["instant_ngp"]["encoding"],
        )
        self.pos_mlp = tcnn.Network(
            self.pos_encoder.n_output_dims,
            16,
            self.config["instant_ngp"]["network"],
        )
        self.dir_encoder = tcnn.Encoding(
            3 + 16 - self.num_density_outputs,
            self.config["instant_ngp"]["dir_encoding"],
        )
        self.dir_mlp = tcnn.Network(
            self.dir_encoder.n_output_dims,
            self.config["num_bands"],
            self.config["instant_ngp"]["rgb_network"],
        )
        self.surf_encoder = tcnn.Encoding(
            2 + 3, self.config["instant_ngp"]["surface_encoding"]
        )
        self.surf_mlp = tcnn.Network(
            self.surf_encoder.n_output_dims,
            self.config["num_bands"],
            self.config["instant_ngp"]["surface_network"],
        )

        self.training = True

        self.max_i = dataset.max_i
        self.loss_fn = {
            "dark": dark_loss,
            "hdr": hdr_loss,
            "l1": l1_loss,
            "l1_plus_hdr": l1_plus_hdr_loss,
            "mse": mse_loss,
            "mse_plus_hdr": mse_plus_hdr_loss,
        }[self.config["loss"].lower()]

    def send_tensors_to(self, device: int) -> None:
        """Move the relevant tensors to a CUDA-capable device.

        Args:
            device: The index of a CUDA-capable device.
        """
        self.device = device

    def get_optimizer(self, config: dict) -> Optimizer:
        """Get the optimizer for this InstantNGPPipeline, which is AdamW with weight
        decay for only the MLP parameters, not the hash table."""
        no_decay_params = chain(
            self.pos_encoder.parameters(),
            self.dir_encoder.parameters(),
            self.surf_encoder.parameters(),
        )
        decay_params = chain(
            self.pos_mlp.parameters(),
            self.dir_mlp.parameters(),
            self.surf_mlp.parameters(),
        )
        optimizer = AdamW(
            [
                {"params": no_decay_params, "weight_decay": 0},
                {"params": decay_params, "weight_decay": config["weight_decay"]},
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
        N = self.config["num_samples_per_ray"]
        pts, z_vals = sample_uniform_bins(
            ray_batch,
            N,
        )
        pts_surf = ray_batch["origin"] + ray_batch["dir"] * ray_batch["len"][:, None]

        if self.point_preprocessor:
            pts = self.point_preprocessor(pts)

        # Instant-NGP uses [0, 1], not [-1, 1]
        pts = (pts + 1) / 2
        pts_surf = (pts_surf + 1) / 2

        # add height above surface to the points vector, if specified in the config
        if self.config["include_height"]:
            pts = append_heights(
                pts, self.ray_origin_height, self.scale, self.offset
            )

        # repeat the direction vector N times to match the points vector
        dirs = ray_batch["dir"][:, None].repeat(1, N, 1)

        # TODO: keep or remove?
        pts[..., 2] = pts[..., 2] / self.config["alt_compress_factor"]

        # apply the MLPs, reshape color output
        pos_enc = self.pos_encoder(pts.view(B_ * N, -1))
        pos_out = self.pos_mlp(pos_enc)
        dir_enc = self.dir_encoder(
            torch.cat(
                [dirs.view(B_ * N, 3), pos_out[:, self.num_density_outputs :]], dim=1
            )
        )
        color = self.dir_mlp(dir_enc)
        color = color.view(B_, N, self.config["num_bands"])

        surf_enc = self.surf_encoder(torch.cat([pts_surf[:, :2], dirs[:, 0]], dim=1))
        color_surf = self.surf_mlp(surf_enc)
        # color_surf = color_surf.view(B_, self.config["num_bands"])

        # pull the densities out of the intermediate output
        sigma = pos_out[..., : self.num_density_outputs].view(B_, N, -1)

        # ReLU activation for color
        color, color_surf = F.relu(color), F.relu(color_surf)

        # ReLU activation for density as it should be non-negative
        sigma = F.relu(sigma)

        # volume rendering
        color_map, _, weights, color_map_atmo, color_map_surf = render_with_surface(
            z_vals * (self.scale / 1000),
            color,
            sigma,
            color_surf,
        )

        results = {
            "color_fine": color[:, :-1],
            "color_surf": color_surf,
            "color_map_surf": color_map_surf,
            "color_map_atmo": color_map_atmo,
            "sigma_fine": sigma[:, :-1],
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
                pts[None], self.ray_origin_height, self.scale, self.offset
            )[0]

        # pts[..., 2] = pts[..., 2] / self.config["alt_compress_factor"]
        pts_comp = torch.cat([pts[..., :2], pts[..., 2:] / self.config["alt_compress_factor"]], dim=-1)

        # pos_out = self.pos_model(pts)
        pos_enc = self.pos_encoder(pts_comp)
        pos_out = self.pos_mlp(pos_enc)

        # the first num_bands values of the intermediate output are treated as densities
        sigma = torch.clip(
            pos_out[..., : self.num_density_outputs].view(
                pts.shape[0], self.num_density_outputs
            ),
            min=0,
        )

        return sigma

    def compute_loss(
        self, ray_batch: Mapping[str, torch.Tensor], results: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute the loss.
        Args:
            ray_batch: A batch of rays and associated observations.
            results: Results from the forward pass of Instant NGP.
        Returns:
            loss: Loss of results with respect to this batch.
        """
        results_indexed = torch.take_along_dim(
            results["color_map_fine"], ray_batch["irgb_idx"][:, None], 1
        )[:, 0]
        gt = ray_batch["rad"].to(dtype=results_indexed.dtype)
        return self.loss_fn(results_indexed, gt, self.max_i)

    def state_dict(self) -> Mapping[str, Mapping[str, Any]]:
        """Get the state dictionary of this InstantNGPPipeline.

        Returns:
            state_dict: Nested dict with all state_dict's in this InstantNGPPipeline.
        """
        state_dict = {
            module_name: getattr(self, module_name).state_dict()
            for module_name in self.module_names
        }
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the weights from a state dictionary into this InstantNGPPipeline.

        Args:
            state_dict: State dict of a InstantNGPPipeline.
        """
        for module_name in self.module_names:
            getattr(self, module_name).load_state_dict(state_dict[module_name])

    def train(self) -> None:
        """Set this InstantNGPPipeline to training mode."""
        self.training = True
        for module_name in self.module_names:
            getattr(self, module_name).train()

    def eval(self) -> None:
        """Set this InstantNGPPipeline to evaluation mode."""
        self.training = False
        for module_name in self.module_names:
            getattr(self, module_name).eval()
