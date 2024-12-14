"""Script containing operations functions for tensors."""


def _prod(iterable: list[float], /, *, start: int = 1) -> float:
    """Return the product of all elements in the iterable."""
    product = start
    for item in iterable:
        product *= item
    return product


def _add(iter1: list[float], iter2: list[float], /) -> list[float]:
    """Return the sum of the elements in the two iterables."""
    assert len(iter1) == len(iter2), "lists must have the same length"
    return [a + b for a, b in zip(iter1, iter2, strict=True)]


def _sub(iter1: list[float], iter2: list[float], /) -> list[float]:
    """Return the difference of the elements in the two iterables."""
    assert len(iter1) == len(iter2), "lists must have the same length"
    return [a - b for a, b in zip(iter1, iter2, strict=True)]


def _mul(iter1: list[float], iter2: list[float], /) -> list[float]:
    """Return the product of the elements in the two iterables."""
    assert len(iter1) == len(iter2), "lists must have the same length"
    return [a * b for a, b in zip(iter1, iter2, strict=True)]


def _div(iter1: list[float], iter2: list[float], /) -> list[float]:
    """Return the quotient of the elements in the two iterables."""
    assert len(iter1) == len(iter2), "lists must have the same length"
    return [a / b for a, b in zip(iter1, iter2, strict=True)]


def _shape(iterable: list[float], /) -> tuple[int]:
    """Return the shape of a list. Supports nested lists."""
    if not iterable:
        return ()

    shape = (len(iterable),)
    while isinstance(iterable[0], list):
        iterable = iterable[0]
        shape = (*shape, len(iterable))
    return shape


def _cast(iterable: list, /, dtype: type) -> list[type]:
    """Return a list with elements cast to the specified type."""
    return [dtype(element) for element in iterable]


if __name__ == "__main__":
    print(__file__)
