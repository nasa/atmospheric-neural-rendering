from atmonr.datasets.harp2 import HARP2Dataset
from atmonr.datasets.harp2_extract import (
    HARP2ExtractDataset,
    HARP2L1CExtractDataset,
    HARP2VoxelGridExtractDataset,
    HARP2GlobalGridExtractDataset,
    HARP2EarthCAREExtractDataset,
)


BANDS = {
    "HARP2": 4,
    # "AirHARP": 4,
}

# add new dataset types to the following lines using the '|' separator, or implement
#   base class if they have any shared functionality
Dataset = HARP2Dataset
ExtractDataset = HARP2ExtractDataset

DATASETS = {
    "HARP2": HARP2Dataset,
    # Not yet implemented: AirHARP
}
EXTRACT_DATASETS = {
    "HARP2": {
        "l1c": HARP2L1CExtractDataset,
        "voxelgrid": HARP2VoxelGridExtractDataset,
        "globalgrid": HARP2GlobalGridExtractDataset,
        "earthcare": HARP2EarthCAREExtractDataset,
    },
    # Not yet implemented: AirHARP
}


def get_dataset(
    config: dict,
    filename: str,
) -> Dataset:
    """Get a Dataset of the provided data type and filename.

    Args:
        config: Config options for the dataset.
        filename: Name of the file containing the satellite scene, which must be present
            in the corresponding `data/...` subdirectory.

    Returns:
        dataset: Dataset corresponding to the provided type and filename.
    """
    if config["type"] not in DATASETS:
        raise NotImplementedError(f"Dataset '{config['type']}' is unrecognized!")
    dataset = DATASETS[config["type"]](
        config=config,
        filename=filename,
    )
    return dataset


def get_extract_dataset(
    mode: str,
    dataset: Dataset,
    *args,
    **kwargs,
) -> ExtractDataset:
    """Get an ExtractDataset which corresponds to an existing Dataset.

    Args:
        mode: The extract mode, which defines the behavior of the ExtractDataset.
        dataset: The dataset from which to extract.

    Returns:
        e_dataset: ExtractDataset for provided Dataset.
    """
    data_type = dataset.config["type"]
    if data_type not in EXTRACT_DATASETS:
        raise NotImplementedError(
            f"ExtractDataset data_type '{data_type}' is unrecognized!"
        )
    e_dataset = EXTRACT_DATASETS[data_type][mode.lower()](
        dataset,
        *args,
        **kwargs,
    )
    return e_dataset
