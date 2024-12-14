"""This module contains the Tensor class, which is the main data structure."""

from typing import Self

from candle.utils.operations import _add, _cast, _div, _mul, _prod, _shape, _sub


class Tensor:
    # pylint: disable=too-few-public-methods

    def __init__(self, data: list, shape: tuple, dtype: type | None = None) -> None:
        """Initialize a tensor instance."""
        self.data = _cast(data, dtype) if dtype else data
        self.dtype = type(self.data[0])
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.size = int(_prod(shape))
        self.strides = self._compute_strides()

    def __add__(self, other: Self) -> Self:
        """Add two tensors element-wise."""
        assert self.shape == other.shape, "Shapes must match for addition"
        return Tensor(_add(self.data, other.data), self.shape)

    def __sub__(self, other: Self) -> Self:
        """Subtract two tensors element-wise."""
        assert self.shape == other.shape, "Shapes must match for subtraction"
        return Tensor(_sub(self.data, other.data), self.shape)
    
    def __mul__(self, other: Self) -> Self:
        """Multiply two tensors element-wise."""
        assert self.shape == other.shape, "Shapes must match for multiplication"
        return Tensor(_mul(self.data, other.data), self.shape)
    
    def __truediv__(self, other: Self) -> Self:
        """"""
        assert self.shape == other.shape, "Shapes must match for multiplication"
        return Tensor(_div(self.data, other.data), self.shape)

    def __len__(self) -> int:
        """Return the size of the tensor."""
        return self.size

    # def __repr__(self) -> str:
    #     """Return the string representation of the tensor."""
    #     data_repr = np.array2string(self.data.reshape(self.shape), separator=", ")
    #     pad = " " * len("Tensor(")
    #     data_repr = data_repr.replace("\n", "\n" + pad)
    #     return f"Tensor({data_repr}, shape={self.shape})"

    # def __str__(self) -> str:
    #     """Return the string representation of the tensor."""
    #     return repr(self)

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
    

# def tensor(data: list, shape: tuple | None = None, dtype: type | None = None) -> Tensor:
#     if isinstance(data, Tensor):
#         data = data.data

#     if dtype is not None:
#         data = _cast(data, dtype)

#     # Parse the shape
#     shape = tuple(shape) if shape is not None else data.shape

#     # Turn to vector if multi-dimensional
#     if data.ndim > 1:
#         data = data.flatten()

#     # Check if the shape is valid
#     assert np.prod(shape) == data.size, "Data size must match the shape"
#     return Tensor(data, shape, dtype)


if __name__ == "__main__":
    a = Tensor([1, 2, 3, 4], (2, 2))
    b = Tensor([5, 6, 7, 8], (2, 2))
    print(a + b)
    print(a - b)
