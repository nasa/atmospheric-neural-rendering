import torch

from atmonr.datasets.harp2 import HARP2Dataset
from atmonr.datasets.harp2_extract import HARP2ExtractDataset


BANDS = {
    "HARP2": 4,
    # "AirHARP": 4,
}

# add new dataset types to the following lines using the '|' separator, or implement
#   base class if they have any shared functionality
Dataset = HARP2Dataset
ExtractDataset = HARP2ExtractDataset

datasets = {
    "HARP2": HARP2Dataset,
    # Not yet implemented: AirHARP
}
extract_datasets = {
    "HARP2": HARP2ExtractDataset,
    # Not yet implemented: AirHARP
}


def get_dataset(
    data_type: str,
    filename: str,
    ray_origin_height: float,
    subsurface_depth: float,
) -> Dataset:
    """Get a Dataset of the provided data type and filename.

    Args:
        data_type: Must be one of the data types present in `datasets`. Determines the
            module to use for parsing the data.
        filename: Name of the file containing the satellite scene, which must be present
            in the corresponding `data/...` subdirectory.
        ray_origin_height: Height above the surface (in meters) at which to construct
            the viewing ray origin points.

    Returns:
        dataset: Dataset corresponding to the provided data_type and filename.
    """
    if data_type not in datasets:
        raise NotImplementedError(f"Dataset '{data_type}' is unrecognized!")
    dataset = datasets[data_type](
        filename=filename,
        ray_origin_height=ray_origin_height,
        subsurface_depth=subsurface_depth,
    )
    return dataset


def get_extract_dataset(
    mode: str,
    data_type: str,
    dataset: Dataset,
    horizontal_step: float,
    sample_alt: torch.Tensor,
) -> ExtractDataset:
    """Get an ExtractDataset which corresponds to an existing Dataset.

    Args:
        mode: The extract mode, which defines the behavior of the ExtractDataset.
        data_type: The data_type for this ExtractDataset.
        dataset: The dataset from which to extract.
        horizontal_step: The horizontal step size in meters between samples.
        sample_alt: The altitudes at which to take samples.

    Returns:
        e_dataset: ExtractDataset for provided Dataset.
    """
    if data_type not in extract_datasets:
        raise NotImplementedError(
            f"ExtractDataset data_type '{data_type}' is unrecognized!"
        )
    e_dataset = extract_datasets[data_type](
        mode=mode,
        dataset=dataset,
        horizontal_step=horizontal_step,
        sample_alt=sample_alt,
    )
    return e_dataset
