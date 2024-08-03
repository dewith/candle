"""This module contains the Tensor class, which is the main data structure."""

import numpy as np
from numpy.typing import ArrayLike


class Tensor:
    """
    A tensor is a multi-dimensional array of numerical values.

    Parameters
    ----------
    data :  array-like
        The data of the tensor.
    shape : tuple
        The shape of the tensor.
    """

    # pylint: disable=too-few-public-methods

    def __init__(self, data: ArrayLike, shape: tuple):
        """
        Initialize a tensor.

        Parameters
        ----------
        data :  array-like
            The data of the tensor.
        shape : tuple
            The shape of the tensor.
        """
        self.data = np.array(data, dtype=np.float32)
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.size = np.prod(shape)
        self.strides = self._compute_strides()
        self.device = "cpu"  # Only CPU is supported since this is Python-only

    def __repr__(self) -> str:
        """
        Return the string representation of the tensor.

        Returns
        -------
        str
            The string representation of the tensor.
        """
        data_repr = np.array2string(self.data, separator=", ")
        pad = " " * len("Tensor(")
        data_repr = data_repr.replace("\n", "\n" + pad)
        return f"Tensor({data_repr}, shape={self.shape})"

    def _compute_strides(self) -> tuple:
        """
        Compute the strides of the tensor.

        The stride of a tensor is the number of elements in the memory between two
        successive elements in the same dimension. The length of the strides array
        have to be equal to the number of dimensions of the tensor.

        Returns
        -------
        tuple
            The strides of the tensor.
        """
        strides = [1]
        for i in reversed(self.shape[1:]):
            strides.insert(0, strides[0] * i)
        return tuple(strides)


def create_tensor(data: ArrayLike, shape: tuple) -> Tensor:
    """
    Create a tensor from data and shape.

    Parameters
    ----------
    data : array-like
        The data of the tensor.
    shape : tuple
        The shape of the tensor.

    Returns
    -------
    Tensor
        A tensor object.
    """
    return Tensor(data, shape)
