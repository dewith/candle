"""Test the tensor creation and manipulation."""

import numpy as np
import torch

import candle


def test_tensor_creation():
    """Test the creation of a tensor."""
    shape = (2, 3, 4, 5)
    data = np.arange(np.prod(shape)).reshape(shape)

    torch_tensor = torch.tensor(data)
    candle_tensor = candle.tensor(data, shape)
    candle_tensor_data = candle_tensor.data.reshape(candle_tensor.shape)

    assert np.allclose(torch_tensor.numpy(), candle_tensor_data)
    assert torch_tensor.shape == candle_tensor.shape
    assert torch_tensor.ndim == candle_tensor.ndim
    assert torch_tensor.stride() == candle_tensor.strides
