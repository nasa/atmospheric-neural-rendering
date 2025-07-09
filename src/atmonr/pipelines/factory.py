from atmonr.datasets.factory import Dataset
from atmonr.pipelines.pipeline import Pipeline
from atmonr.pipelines.instant_ngp import InstantNGPPipeline
from atmonr.pipelines.nerf import NeRFPipeline


_PIPELINES = {
    "NeRF": NeRFPipeline,
    "InstantNGP": InstantNGPPipeline,
}


def get_pipeline(config: dict, dataset: Dataset) -> Pipeline:
    """Get a neural rendering Pipeline.

    Args:
        config: Configuration options for the Pipeline.
        dataset: Dataset to which the Pipeline will be applied.

    Returns:
        pipeline: A Pipeline, of type defined by the provided configuration.
    """
    pipeline_type = config["type"]
    if pipeline_type not in _PIPELINES:
        raise NotImplementedError(f"Pipeline '{pipeline_type}' is unrecognized!")
    pipeline = _PIPELINES[pipeline_type](config, dataset)
    return pipeline
