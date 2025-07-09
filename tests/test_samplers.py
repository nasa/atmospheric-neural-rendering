import numpy as np
import pytest
import torch

from atmonr.samplers import sample_uniform_bins


# fake batch to use for testing
@pytest.fixture(scope="module")
def batch() -> dict[str, torch.Tensor]:
    og = torch.from_numpy(
        np.mgrid[-1:1.01:0.1, -1:1.01:0.1, -1:1.01:0.1].astype(np.float32)
    )
    og = og.reshape(3, -1).T
    return {
        "origin": og,
        "dir": -og,
        "len": torch.zeros(og.shape[0]) + 2,
    }


# test that sample_uniform_bins behaves as expected
def test_sample_uniform_bins(batch) -> None:
    torch.manual_seed(6558903984)  # for reproducibility
    pts, z_vals = sample_uniform_bins(batch)
    assert (pts >= -1).all() and (pts <= 1).all()
    assert (z_vals >= 0).all() and (z_vals <= 2).all()
    # TO DO: more tests


# TO DO: test sample_pdf
# def test_sample_pdf() -> None:
#     pass
