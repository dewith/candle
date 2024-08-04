"""This module contains the Tensor class, which is the main data structure."""

from typing import Self

import numpy as np
import numpy.typing as npt


class Tensor:
    """
    A tensor is a multi-dimensional array of numerical values.

    Parameters
    ----------
    data :  array-like
        The data of the tensor. Should be a 1D array.
    shape : tuple
        The shape of the tensor. Should be a tuple of integers.
    dtype : dtype
        The data type of the tensor.
        Receives objects that can be coerced into a dtype. This includes the likes of:
        - Native type objects.
        - Character codes or the names of type objects.
        - Objects with the .dtype attribute.
    """

    # pylint: disable=too-few-public-methods

    def __init__(
        self, data: npt.ArrayLike, shape: tuple, dtype: npt.DTypeLike = None
    ):  # numpydoc ignore=PR01
        """Initialize a tensor instance."""
        self.data = np.array(data, dtype=dtype)
        self.dtype = self.data.dtype
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.size = int(np.prod(shape))
        self.strides = self._compute_strides()

    def __add__(self, other: Self) -> Self:  # numpydoc ignore=PR01
        """Add two tensors element-wise."""
        assert self.shape == other.shape, "Shapes must match for addition"
        result = self.data + other.data
        return Tensor(result, self.shape)

    def __sub__(self, other: Self) -> Self:  # numpydoc ignore=PR01
        """Subtract two tensors element-wise."""
        assert self.shape == other.shape, "Shapes must match for subtraction"
        result = self.data - other.data
        return Tensor(result, self.shape)

    def __mul__(self, other: Self) -> Self:  # numpydoc ignore=PR01
        """Multiply two tensors element-wise."""
        assert self.shape == other.shape, "Shapes must match for multiplication"
        result = self.data * other.data
        return Tensor(result, self.shape)

    def __getitem__(self, key: int | slice | tuple) -> Self:  # numpydoc ignore=PR01
        """Get the item at the specified index."""
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) > self.ndim:
            raise IndexError(f"Too many indices for tensor of dimension {self.ndim}")

        working_array = self.data.copy()
        newshape = self.shape
        for i, k in enumerate(key):
            dim_size = self.shape[i]
            dim_stride = self.strides[i]

            # Get start and end indices
            if isinstance(k, int):
                if k < 0:
                    k += dim_size
                start, end = k, k + 1
            elif isinstance(k, slice):
                start = 0 if k.start is None else k.start
                end = dim_size if k.stop is None else min(k.stop, dim_size)
            else:
                raise TypeError(f"Invalid key type: {type(k)}")

            # Handle negative indices
            start = start + dim_size if start < 0 else start
            end = end + dim_size if end < 0 else end

            # Check bounds
            idx_err_msg = "Index {} is out of bounds for dimension {} with size {}"
            if not 0 <= start < dim_size:
                raise IndexError(idx_err_msg.format(start, i, dim_size))
            if not 0 <= end <= dim_size:
                raise IndexError(idx_err_msg.format(end, i, dim_size))

            # Compute new shape
            slice_len = end - start
            if slice_len == 0:
                newshape = (0,)
            elif slice_len > 1:
                newshape = (slice_len,) + self.shape[i + 1 :]
            else:
                newshape = self.shape[i + 1 :]

            # Apply slice
            start *= dim_stride
            end *= dim_stride
            working_array = working_array[start:end]
        return Tensor(working_array, newshape)

    def __len__(self) -> int:
        """Return the size of the tensor."""
        return self.size

    def __repr__(self) -> str:
        """Return the string representation of the tensor."""
        data_repr = np.array2string(self.data.reshape(self.shape), separator=", ")
        pad = " " * len("Tensor(")
        data_repr = data_repr.replace("\n", "\n" + pad)
        return f"Tensor({data_repr}, shape={self.shape})"

    def __str__(self) -> str:
        """Return the string representation of the tensor."""
        return repr(self)

    def __copy__(self) -> Self:
        """Return a shallow copy of the tensor."""
        return Tensor(self.data.copy(), self.shape)

    def _compute_strides(self) -> tuple:
        """
        Compute the strides of the tensor.

        Notes
        -----
        The stride of a tensor is the number of elements in the memory between two
        successive elements in the same dimension. The length of the strides array
        have to be equal to the number of dimensions of the tensor.

        This method is highly efficient and is used to compute the strides of the tensor
        when it is created. It is not meant to be called by the user.
        """
        strides = [1]
        for i in reversed(self.shape[1:]):
            strides.insert(0, strides[0] * i)
        return tuple(strides)


def tensor(
    data: npt.ArrayLike, shape: tuple = None, dtype: npt.DTypeLike = None
) -> Tensor:
    """
    Create a tensor from the given data and shape.

    Parameters
    ----------
    data :  array-like
        The data of the tensor.
    shape : tuple
        The shape of the tensor. Should be a tuple of integers.
        If not provided, it is inferred from the data.
    dtype : dtype
        The data type of the tensor.
        Receives objects that can be coerced into a dtype. This includes the likes of:
        - Native type objects.
        - Character codes or the names of type objects.
        - Objects with the .dtype attribute.
        If not provided, it is inferred from the data.

    Returns
    -------
    Tensor
        The created tensor.
    """
    # Parse the data
    if isinstance(data, Tensor):
        data = data.data

    dtype = np.dtype(dtype) if dtype is not None else np.result_type(*data)
    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=dtype)

    # Parse the shape
    shape = tuple(shape) if shape is not None else data.shape

    # Turn to vector if multi-dimensional
    if data.ndim > 1:
        data = data.flatten()

    # Check if the shape is valid
    assert np.prod(shape) == data.size, "Data size must match the shape"
    return Tensor(data, shape, dtype)
