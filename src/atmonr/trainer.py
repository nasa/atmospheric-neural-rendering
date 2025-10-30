from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from atmonr.batch_loader import BatchLoader
from atmonr.datasets.factory import Dataset
from atmonr.pipelines.pipeline import Pipeline
from atmonr.utils import dict_to


class Trainer:
    """Trains a neural rendering pipeline on a given dataset."""

    def __init__(
        self, config: dict, dataset: Dataset, pipeline: Pipeline, exp_name: str
    ) -> None:
        """Initialize a Trainer.

        Args:
            config: Training configuration options.
            dataset: Dataset of observed radiances and geometry.
            pipeline: Neural rendering pipeline to be trained.
            exp_name: Name of this experiment.
        """
        self.config = config
        self.dataset = dataset
        self.pipeline = pipeline
        self.device = torch.cuda.current_device()

        if config["all_gpu"]:
            assert config["num_workers"] == 0
            self.dataloader = BatchLoader(
                dataset,
                batch_size=config["batch_size"],
                shuffle=True,
            )
        else:
            self.dataloader = DataLoader(
                dataset,
                num_workers=config["num_workers"],
                batch_size=config["batch_size"],
                shuffle=True,
            )

        self.epoch_idx = 0
        self.iter_count = 0
        self.num_epochs = int(-(self.config["num_iters"] // -len(self.dataloader)))
        self.optimizer = pipeline.get_optimizer(self.config["optimizer"])

        if self.config["scheduler"]["type"] == "target_lr":
            gamma = gamma = (
                self.config["scheduler"]["final_lr"] / self.config["optimizer"]["lr"]
            ) ** (1 / self.num_epochs)
            self.scheduler = ExponentialLR(optimizer=self.optimizer, gamma=gamma)
        elif self.config["scheduler"]["type"] == "fixed":
            self.scheduler = ExponentialLR(
                optimizer=self.optimizer, gamma=self.config["scheduler"]["gamma"]
            )
        else:
            raise NotImplementedError(
                f"Unknown scheduler type {self.config['scheduler']['type']}"
            )

        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tensorboard_dir = Path("data") / "tensorboard" / f"{exp_name}_{now_str}"
        self.writer = SummaryWriter(self.tensorboard_dir)

    def train(self, output_path: Path, profile: bool = False) -> None:
        """Train the pipeline on the provided dataset.

        Args:
            output_path: Path to the directory where checkpoints will be saved.
            profile: Whether to use profiling mode to analyze code performance.
        """

        prof = None
        if profile:
            prof = self.get_profiler()
            prof.start()

        progress = self.dataset.get_progress_tracker()

        last_update_len = 0
        running_losses = []
        while self.iter_count < self.config["num_iters"]:
            for batch in self.dataloader:
                if prof:
                    prof.step()

                # move the batch's tensors to GPU
                if not self.config["all_gpu"]:
                    batch = dict_to(batch, self.device)

                results = self.pipeline.forward(batch)

                # get the loss, apply it, and take an optimization step
                loss = self.pipeline.compute_loss(batch, results)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # update tensorboard with the loss function
                self.writer.add_scalar("Loss", loss.item(), self.iter_count)
                running_losses = running_losses[-self.config["print_frequency"] :] + [
                    loss.item()
                ]
                self.iter_count += 1
                # step the scheduler if it's of type fixed
                if (
                    self.config["scheduler"]["type"] == "fixed"
                    and self.iter_count % self.config["scheduler"]["decay_interval"]
                    == 0
                    and self.iter_count > self.config["scheduler"]["decay_start"]
                ):
                    self.scheduler.step()

                # update the progress tracker
                pred_pixels = torch.take_along_dim(
                    results["color_map_fine"], batch["irgb_idx"][:, None], dim=1
                )[:, 0]
                progress.pred_pixels[batch["idx"].cpu().numpy()] = (
                    pred_pixels.detach().cpu().float().numpy()
                )
                pred_pixels_surf = torch.take_along_dim(
                    results["color_map_surf"], batch["irgb_idx"][:, None], dim=1
                )[:, 0]
                progress.pred_pixels_surf[batch["idx"].cpu().numpy()] = (
                    pred_pixels_surf.detach().cpu().float().numpy()
                )
                pred_pixels_atmo = torch.take_along_dim(
                    results["color_map_atmo"], batch["irgb_idx"][:, None], dim=1
                )[:, 0]
                progress.pred_pixels_atmo[batch["idx"].cpu().numpy()] = (
                    pred_pixels_atmo.detach().cpu().float().numpy()
                )

                # quit if enough iters
                if self.iter_count >= self.config["num_iters"]:
                    break
                # report some statistics every so many iters
                if (self.iter_count % self.config["print_frequency"]) == 0:
                    mean_loss = sum(running_losses) / len(running_losses)
                    # update line with running losses
                    update_line = (
                        f"{self.iter_count}/{self.config['num_iters']} | "
                        f"Loss: {mean_loss:.5f}"
                    )
                    update_len = len(update_line)
                    print(
                        update_line + max(0, last_update_len - update_len) * " ",
                        end="\r",
                    )
                    last_update_len = update_len

            # update the progress image after the whole image has been processed
            progress.pred_img[progress.valid] = progress.pred_pixels
            pred_img = torch.from_numpy(progress.pred_img).to(self.pipeline.device)
            target_img = torch.from_numpy(progress.target_img).to(self.pipeline.device)
            target_img[target_img.isnan()] = 0
            pred_img = pred_img.permute(2, 0, 1)
            target_img = target_img.permute(2, 0, 1)
            progress.pred_img_surf[progress.valid] = progress.pred_pixels_surf
            pred_img_surf = torch.from_numpy(progress.pred_img_surf).to(
                self.pipeline.device
            )
            pred_img_surf = pred_img_surf.permute(2, 0, 1)
            progress.pred_img_atmo[progress.valid] = progress.pred_pixels_atmo
            pred_img_atmo = torch.from_numpy(progress.pred_img_atmo).to(
                self.pipeline.device
            )
            pred_img_atmo = pred_img_atmo.permute(2, 0, 1)

            self.epoch_idx += 1

            # step learning rate scheduler if it's of type target_lr
            if self.config["scheduler"]["type"] == "target_lr":
                self.scheduler.step()

            # print and update tensorboard with image similarity metrics
            image_metrics = self.dataset.get_image_metrics(pred_img, target_img)
            update_line = f"Epoch {self.epoch_idx}/{self.num_epochs}"
            for metric_name, metric_val in image_metrics.items():
                if isinstance(metric_val, list):
                    continue
                update_line += f" | {metric_name}: {metric_val:.3f}"
                self.writer.add_scalar(metric_name, metric_val, self.epoch_idx)
            update_len = len(update_line)
            print(update_line + max(0, last_update_len - update_len) * " ")
            last_update_len = update_len

            # update tensorboard with side-by-side image comparison
            pred_img_rgb = self.dataset.get_rgb(pred_img).cpu().numpy()
            pred_img_surf_rgb = self.dataset.get_rgb(pred_img_surf).cpu().numpy()
            pred_img_atmo_rgb = self.dataset.get_rgb(pred_img_atmo).cpu().numpy()
            viz = np.concatenate(
                [
                    pred_img_surf_rgb,
                    pred_img_atmo_rgb,
                    pred_img_rgb,
                    progress.target_img_rgb,
                ],
                axis=1,
            )
            self.writer.add_image(
                f"Epoch {self.epoch_idx}", np.transpose(viz, (2, 0, 1))
            )

            # checkpoint every epoch
            self.save(output_path, self.epoch_idx)

            # stop profiler after the first epoch
            if prof:
                prof.stop()
        print()

    def get_profiler(self) -> torch.profiler.profile:
        """Get a profiler for this trainer."""
        return torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=10, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                str(self.tensorboard_dir)
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=True,
        )

    def save(self, output_path: Path, epoch: int) -> None:
        """Save the pipeline and optimizer from this training run to file.

        Args:
            output_path: Path to the directory where the checkpoint will be saved.
            epoch: Current epoch.
        """
        torch.save(
            {
                "pipeline": self.pipeline.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "tensorboard_dir": self.tensorboard_dir,
                "epoch_idx": self.epoch_idx,
                "iter_count": self.iter_count,
            },
            output_path / f"epoch_{epoch:04d}.pt",
        )

    def load(self, output_path: Path) -> None:
        """Load a Trainer from checkpoint. Will load the state_dict for this Trainer's
        Pipeline, optimizer, and scheduler, as well as the TensorBoard directory.

        Args:
            output_path: Path to the directory with the checkpoint to load.
        """
        ckpts = list(output_path.glob("epoch_*.pt"))
        last_ckpt_path = sorted(ckpts, key=lambda c: int(c.stem.split("_")[1]))[-1]
        last_ckpt = torch.load(last_ckpt_path)
        self.pipeline.load_state_dict(last_ckpt["pipeline"])
        self.optimizer.load_state_dict(last_ckpt["optimizer"])
        self.scheduler.load_state_dict(last_ckpt["scheduler"])
        self.tensorboard_dir = last_ckpt["tensorboard_dir"]
        self.writer = SummaryWriter(self.tensorboard_dir)
        self.epoch_idx = last_ckpt["epoch_idx"]
        self.iter_count = last_ckpt["iter_count"]
