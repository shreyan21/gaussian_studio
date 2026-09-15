"""Tiny tensors validate the real upstream coordinate conversion, without claiming
that the large pretrained network has run on this CPU-only computer.
"""
import sys

import numpy as np
import pytest

from studio.config import SHARP_SOURCE
from studio.worker import filter_pixel_gaussians


def test_foreground_mask_keeps_all_sharp_layers_in_pixel_order():
    gaussians = np.arange(16 * 8, dtype=np.float32).reshape(8, 16)
    mask = np.array([[True, False], [False, True]])
    filtered, layers = filter_pixel_gaussians(gaussians, mask)
    assert layers == 2
    np.testing.assert_array_equal(filtered, gaussians[[0, 3, 4, 7]])


def test_foreground_mask_rejects_mismatched_output():
    with pytest.raises(RuntimeError, match="does not match"):
        filter_pixel_gaussians(np.zeros((7, 16), dtype=np.float32), np.ones((2, 2), dtype=bool))


@pytest.mark.skipif(not (SHARP_SOURCE / "sharp" / "models" / "__init__.py").is_file(), reason="Optional SHARP source not installed")
def test_upstream_unprojection_and_viewer_quaternion_convention():
    import torch
    sys.path.insert(0,str(SHARP_SOURCE))
    from sharp.utils.gaussians import Gaussians3D, unproject_gaussians, compose_covariance_matrices
    g=Gaussians3D(mean_vectors=torch.tensor([[[.2,.4,2.]]]),singular_values=torch.tensor([[[.1,.2,.3]]]),quaternions=torch.tensor([[[1.,0.,0.,0.]]]),colors=torch.ones((1,1,3))*.5,opacities=torch.ones((1,1))*.8)
    k=torch.tensor([[768.,0.,768.,0.],[0.,768.,768.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]])
    converted=unproject_gaussians(g,torch.eye(4),k,(1536,1536))
    np.testing.assert_allclose(converted.mean_vectors.numpy(),g.mean_vectors.numpy(),atol=1e-6)
    q=converted.quaternions.reshape(-1,4).numpy()
    rotated=np.stack((-q[:,1],q[:,0],-q[:,3],q[:,2]),axis=-1)
    covariance=compose_covariance_matrices(converted.quaternions,converted.singular_values).numpy()
    viewer_cov=compose_covariance_matrices(torch.from_numpy(rotated)[None],converted.singular_values).numpy()
    rotation=np.diag([1,-1,-1])
    np.testing.assert_allclose(viewer_cov,rotation@covariance@rotation,atol=1e-6)
    assert np.linalg.eigvalsh(viewer_cov).min()>0
