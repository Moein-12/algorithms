from collections import abc
import typing as tp


T = tp.TypeVar('T')


class SupportsCompetitions(tp.Protocol):
    """Protocol for types that support the less than operator (<)
    and greater than operator (>)."""
    def __lt__(self, other: tp.Self) -> bool: ...
    def __gt__(self, other: tp.Self) -> bool: ...

Key: tp.TypeAlias = abc.Callable[[T], SupportsCompetitions]
"""Type alias for a key function that extracts a comparison key from elements.
This type alias represents a function that takes an element and returns a value
that can be compared using the less-than operator (<) to their own kind.
"""

UNDEFINED: tp.Final[tp.Any] = object()  # used as default value of iterations


def identity(x: tp.Any) -> tp.Any:
    """Identity function that returns its input untouched.
    
    Args:
        x: Any value.
    
    Returns:
        The input value untouched.
    """
    return x

def positive_index(index: int | None, length: int) -> int:  # to turn pythonic indexes to algorithmic ones
    """Converts a Python-style index to a positive index.
    
    Args:
        index: The index to convert or None.
        length: The length of the sequence.
    
    Returns:
        int: The positive index or the length if index is None.
        
    Raises:
        IndexError: If the index is out of range.
    """
    if index is None:
        return length

    if index < -length or length <= index:
        raise IndexError("index out of range")

    return index + length if index < 0 else index
