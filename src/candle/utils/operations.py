"""Script containing operations functions for tensors."""

from collections.abc import Iterable
from typing import List


def prod(iterable: Iterable[float], /, *, start: int = 1) -> float:
    """Return the product of all elements in the iterable."""
    product = start
    for item in iterable:
        product *= item
    return product


def add(iterable: Iterable[float], iterable2: Iterable[float], /) -> List[float]:
    """Return the sum of the elements in the two iterables."""
    assert len(iterable) == len(iterable2), "Iterables must have the same length"
    return [a + b for a, b in zip(iterable, iterable2)]


def sub(iterable: Iterable[float], iterable2: Iterable[float], /) -> List[float]:
    """Return the difference of the elements in the two iterables."""
    assert len(iterable) == len(iterable2), "Iterables must have the same length"
    return [a - b for a, b in zip(iterable, iterable2)]


def mul(iterable: Iterable[float], iterable2: Iterable[float], /) -> List[float]:
    """Return the product of the elements in the two iterables."""
    assert len(iterable) == len(iterable2), "Iterables must have the same length"
    return [a * b for a, b in zip(iterable, iterable2)]


def div(iterable: Iterable[float], iterable2: Iterable[float], /) -> List[float]:
    """Return the quotient of the elements in the two iterables."""
    assert len(iterable) == len(iterable2), "Iterables must have the same length"
    return [a / b for a, b in zip(iterable, iterable2)]


def shape(iterable: list[float], /) -> tuple[int]:
    """Return the shape of a list. Supports nested lists."""
    if not iterable:
        return ()
    return (len(iterable),) + shape(iterable[0])
