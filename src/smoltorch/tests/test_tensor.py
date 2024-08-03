"""Test the tensor creation and manipulation."""

import numpy as np
import torch

from smoltorch import tensor


def test_tensor_creation():
    """Test the creation of a tensor."""
    shape = (2, 3, 4, 5)
    data = np.arange(np.prod(shape)).reshape(shape)

    torch_tensor = torch.tensor(data)
    smol_tensor = tensor.Tensor(data, shape)

    assert np.allclose(torch_tensor.numpy(), smol_tensor.data)
    assert torch_tensor.shape == smol_tensor.shape
    assert torch_tensor.ndim == smol_tensor.ndim
    assert torch_tensor.stride() == smol_tensor.strides
