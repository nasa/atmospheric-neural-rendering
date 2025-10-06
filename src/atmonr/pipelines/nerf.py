from itertools import chain
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch.optim import Adam, Optimizer

from atmonr.datasets.factory import Dataset
from atmonr.encoders import positional_encoding
from atmonr.graphics_utils import render
from atmonr.models.nerf import get_model
from atmonr.pipelines.pipeline import Pipeline
from atmonr.samplers import append_heights, sample_pdf, sample_uniform_bins


class NeRFPipeline(Pipeline):
    """This pipeline uses an approach based on Neural Radiance Fields (NeRF) to render
    an atmospheric scene.

    NeRF paper: https://arxiv.org/abs/2003.08934
    """

    def __init__(
        self,
        config: dict,
        dataset: Dataset,
    ) -> None:
        """Initialize a NeRFPipeline.

        Args:
            config: Configuration options for this NeRFPipeline.
            dataset: Dataset to which this pipeline will be applied.
        """
        super().__init__(config, dataset)

        self.nerf = {}
        self.nerf["coarse"], self.nerf["fine"] = get_model(
            hidden_dim=config["mlp_hidden_dim"],
            N_lambda=config["num_bands"],
            L_x=config["encoder"]["L_x"],
            L_d=config["encoder"]["L_d"],
            include_height=config["include_height"],
        )
        self.training = True

    def send_tensors_to(self, device: int) -> None:
        """Move the relevant tensors to a CUDA-capable device.

        Args:
            device: The index of a CUDA-capable device.
        """
        self.device = device
        self.nerf["coarse"] = self.nerf["coarse"].to(device)
        self.nerf["fine"] = self.nerf["fine"].to(device)

    def get_optimizer(
        self,
        config: dict,
    ) -> Optimizer:
        """Get the Adam optimizer for this NeRF pipeline.

        Args:
            config: Configuration options for the optimizer.
            num_epochs: Number of epochs over which to train.

        Returns:
            optimizer: An optimizer of the parameters in this pipeline.
        """
        params = chain(self.nerf["coarse"].parameters(), self.nerf["fine"].parameters())
        optimizer = Adam(params=params, lr=config["lr"])
        return optimizer

    def _forward(
        self,
        mode: str,
        ray_batch: Mapping[str, torch.Tensor],
        weights_coarse: None | torch.Tensor = None,
        z_vals_coarse: None | torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        """Run either the 'coarse' or 'fine' stage of the NeRF forward pass. The coarse
        stage uses uniform sampling of stratified bins along the viewing rays, while the
        fine stage uses the coarse model's (normalized) density output as a probability
        density function from which to sample, in addition to the original coarse
        samples. For more details, read the NeRF paper.

        Args:
            mode: Whether to run the 'coarse' or 'fine' stage of NeRF.
            ray_batch: A batch of rays and associated observations.
            weights_coarse: The density outputs of the coarse model, needed only in
                fine mode.
            z_vals_coarse: The values along a viewing ray of the coarse samples, needed
                only in fine mode.

        Returns:
            results: Partial NeRF results, from either the 'coarse' or 'fine' stage of
                the forward pass.
        """
        assert (mode == "coarse" and z_vals_coarse is None) or (
            mode == "fine" and z_vals_coarse is not None
        )

        # get some shorter names for array shapes
        B_ = ray_batch["origin"].shape[0]
        L_x = self.config["encoder"]["L_x"]
        L_d = self.config["encoder"]["L_d"]

        # sample uniformly along rays for the coarse NeRF
        if mode == "coarse":
            N = self.config["sampler"]["N_c"]
            pts, z_vals = sample_uniform_bins(ray_batch, n_bins=N)
        else:
            assert isinstance(weights_coarse, torch.Tensor)
            assert isinstance(z_vals_coarse, torch.Tensor)
            N = self.config["sampler"]["N_c"] + self.config["sampler"]["N_f"]
            pts, z_vals = sample_pdf(
                ray_batch,
                weights_coarse,
                z_vals_coarse,
                n_samples=self.config["sampler"]["N_f"],
            )

        # if we have a point preprocessing function
        if self.point_preprocessor:
            pts = self.point_preprocessor(pts)

        # add height above surface to the points vector, if specified in the config
        if self.config["include_height"]:
            pts = append_heights(pts, self.ray_origin_height, self.scale, self.offset)

        # position encoding with L_x frequencies for the points vector
        pts_enc = positional_encoding(pts, L_x).view((B_ * N, -1))
        # repeat the direction vector N_c times to match the points vector
        dirs = ray_batch["dir"][:, None].repeat(1, N, 1)
        # position encoding with L_d frequencies for the direction vector
        dirs_enc = positional_encoding(dirs, L_d).view((B_ * N, -1))

        # concatenate points and directions to get input to model
        x = torch.cat([pts_enc, dirs_enc], dim=1)

        # run the model, reshape outputs
        color, sigma = self.nerf[mode](x)
        color = color.view(B_, N, -1)
        # density is per-band in the fine model output
        if mode == "coarse":
            sigma = sigma.view(B_, N, 1)
        else:
            sigma = sigma.view(B_, N, -1)

        # exponential activation for color, clamp to 11 to avoid overflow w/ float16
        color = torch.exp(torch.clamp(color, max=11))

        # ReLU activation for density as it should be non-negative
        sigma = F.relu(sigma)

        # volume rendering
        color_map, _, weights = render(z_vals * (self.scale / 1000), color, sigma)

        results = {
            f"color_{mode}": color,
            f"sigma_{mode}": sigma,
            f"color_map_{mode}": color_map,
            f"weights_{mode}": weights,
            f"z_vals_{mode}": z_vals,
        }
        if self.config["include_height"]:
            results[f"norm_heights_{mode}"] = pts[..., 3]
        return results

    def forward(self, ray_batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run both the coarse and fine stages of NeRF and return the results.

        Args:
            ray_batch: A batch of rays and associated observations.

        Returns:
            results: Aggregated results from both the coarse and fine stages of the NeRF
                forward pass.
        """
        results = self._forward("coarse", ray_batch)
        results.update(
            self._forward(
                "fine",
                ray_batch,
                weights_coarse=results["weights_coarse"],
                z_vals_coarse=results["z_vals_coarse"],
            )
        )
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
        if self.config["include_height"]:
            pts = append_heights(
                pts[None], self.ray_origin_height, self.scale, self.offset
            )[0]
        # position encoding with L_x frequencies for the points vector
        pts_enc = (
            positional_encoding(pts, self.config["encoder"]["L_x"])
            .view(pts.shape[0], -1)
            .float()
        )
        _, sigma = self.nerf["fine"].forward_pos_only(pts_enc)
        sigma = torch.clip(sigma, min=0)
        return sigma

    def compute_loss(
        self, ray_batch: Mapping[str, torch.Tensor], results: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute the combined coarse and fine MSE losses.

        Args:
            ray_batch: A batch of rays and associated observations.
            results: Aggregated results from both the coarse and fine stages of the NeRF
                forward pass.
        Returns:
            loss: Sum of the MSE losses for both the coarse and fine models.
        """
        results_c = torch.take_along_dim(
            results["color_map_coarse"], ray_batch["irgb_idx"][:, None], 1
        )[:, 0]
        results_f = torch.take_along_dim(
            results["color_map_fine"], ray_batch["irgb_idx"][:, None], 1
        )[:, 0]
        loss_c = F.mse_loss(results_c, ray_batch["rad"])
        loss_f = F.mse_loss(results_f, ray_batch["rad"])
        loss = loss_c + loss_f
        return loss

    def state_dict(self) -> Mapping[str, Mapping[str, Any]]:
        """Get the state dictionary of this NeRFPipeline.

        Returns:
            state_dict: Nested dict with all state_dict's in this NeRFPipeline.
        """
        state_dict = {
            "coarse": self.nerf["coarse"].state_dict(),
            "fine": self.nerf["fine"].state_dict(),
        }
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the weights from a state dictionary into this NeRFPipeline.

        Args:
            state_dict: State dict of a NeRFPipeline.
        """
        self.nerf["coarse"].load_state_dict(state_dict["coarse"])
        self.nerf["fine"].load_state_dict(state_dict["fine"])

    def train(self) -> None:
        """Set this NeRFPipeline to training mode."""
        self.training = True
        self.nerf["coarse"].train()
        self.nerf["fine"].train()

    def eval(self) -> None:
        """Set this NeRFPipeline to evaluation mode."""
        self.training = False
        self.nerf["coarse"].eval()
        self.nerf["fine"].eval()
