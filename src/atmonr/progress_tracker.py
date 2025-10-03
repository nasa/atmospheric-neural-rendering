from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class ProgressTracker:
    """Tracks progress during training. Is not used to compute the loss function, but is
    used to compute reconstruction metrics and for visualization purposes.
    """

    # mask of valid coordinates in the imagery
    valid: npt.NDArray[np.bool_]
    # ground-truth (all locations)
    target_img: npt.NDArray[np.float64]
    # ground-truth, RGB image (nadir view)
    target_img_rgb: npt.NDArray[np.float64]
    # predicted image (all locations)
    pred_img: npt.NDArray[np.float64]
    # predicted pixels (only valid locations)
    pred_pixels: npt.NDArray[np.float64]
    # surface
    pred_img_surf: npt.NDArray[np.float64]
    pred_pixels_surf: npt.NDArray[np.float64]
    # atmosphere
    pred_img_atmo: npt.NDArray[np.float64]
    pred_pixels_atmo: npt.NDArray[np.float64]
