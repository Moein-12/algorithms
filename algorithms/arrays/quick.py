"""This module provides various implementations of quicksort algorithms
and related utilities.

### Main functions:
- `quick_sort()`: the main function that sorts the sequence
- `quick_select()`: the main function that selects the k-th smallest element
- `validate_pivot()`: validates the pivot selector function. not included in
  public API because it takes a lot of time to run.

### Other Public Functions:
- partition functions
    - `partition()`: the function that partitions the sequence with one pivot
    - `multi_pivot_partition()`: the function that partitions the sequence with
- Pivot selection strategies
    - `first_as_pivot()`: Returns first index as pivot
    - `last_as_pivot()`: Returns last index as pivot  
    - `middle_as_pivot()`: Returns middle index as pivot
    - `random_pivot()`: Returns random index as pivot
    - `first_n_items()`: Returns first n indices as pivots
    - `last_n_items()`: Returns last n indices as pivots
    - `n_random_pivots()`: Returns n random indices as pivots

Private Functions:
these are implemented considering inputs are valid and indexes are
non-negative.
- `_quick_sort()`: the function that contains the recursive quicksort
- `_iterative_quick_sort()`: contains the iterative quicksort
- `_multi_pivot_quicksort()`: the function that contains the recursive
  multi-pivot quicksort
- `_iterative_multi_pivot_quicksort()`: contains the iterative version

Check the functions' documentations for more information.
"""

from collections import abc
from itertools import pairwise
import typing as tp
import random as rnd

from ._utilities import identity, positive_index, Key


T = tp.TypeVar('T')
T2 = tp.TypeVar('T2')


# predefined pivot functions for parameterization

# single pivot selection

def first_as_pivot(length: int) -> int:
    """Returns the first index (0) as the pivot.
    
    Args:
        length: The length of the sequence.
    
    Returns:
        int: Always returns 0 as the pivot index.
    """
    return 0

def last_as_pivot(length: int) -> int:
    """Returns the last index as the pivot.
    
    Args:
        length: The length of the sequence.
    
    Returns:
        int: The last valid index (length - 1).
    """
    return length - 1  # since length is exclusive

def middle_as_pivot(length: int) -> int:
    """Returns the middle index as the pivot.
    
    Args:
        length: The length of the sequence.
    
    Returns:
        int: The middle index (length // 2).
    """
    return (length) >> 1  # equivalent to length // 2

def random_pivot(length: int) -> int:
    """Returns a random index as the pivot.
    
    Args:
        length: The length of the sequence.
    
    Returns:
        int: A random index between 0 and length-1.
    """
    return rnd.randrange(0, length)


# multi-pivot selection

def first_n_items(length: int, n: int) -> list[int]:
    """Returns the first n indices as pivots.
    
    Args:
        length: The length of the sequence.
        n: Number of pivot indices to return.
    
    Returns:
        list[int]: List of the first n indices.
    """
    return list(range(n))

def last_n_items(length: int, n: int) -> list[int]:
    """Returns the last n indices as pivots.
    
    Args:
        length: The length of the sequence.
        n: Number of pivot indices to return.
    
    Returns:
        list[int]: List of the last n indices.
    """
    return list(range(length - n, length))

def n_random_pivots(length: int, n: int) -> list[int]:
    """Returns n random unique indices as pivots.
    
    Args:
        length: The length of the sequence.
        n: Number of pivot indices to return.
    
    Returns:
        list[int]: Sorted list of n unique random indices.
    """
    return sorted(rnd.sample(range(length), n))  # returns n unique random numbers in the range of length

def divider_pivots(length: int, n: int) -> list[int]:
    """Returns n-1 evenly spaced indices as pivots.
    
    Args:
        length: The length of the sequence.
        n: Number of sections to divide the sequence into (resulting in n-1 pivots).
    
    Returns:
        list[int]: List of n-1 evenly spaced indices.
    """
    return [i * length // n for i in range(n)]  # returns n numbers in the range of length


# bonus (there's no internal validation in quick_sort and quick_select APIs)
def validate_pivot_selector(max_length: int, pivot: int) -> list[int]:
    """Validates a pivot selector by checking all possible sequence lengths.
    
    Args:
        max_length: Maximum length to check.
        pivot: Pivot index to validate.
    
    Returns:
        list[int]: List of sequence lengths where the pivot is invalid.
    """
    errors = []
    for i in range(max_length):
        if pivot >= i:
            errors.append(i)
    return errors


# single-pivot quicksort

def partition(  # Hoare's algorithm
    seq: abc.MutableSequence[T],
    low: int,
    high: int,
    pivot_index: int,
    key: Key[T] = identity,
) -> int:
    """Partitions a sequence around a pivot using Hoare's algorithm.
    
    Args:
        seq: The sequence to partition.
        low: The lower bound index.
        high: The upper bound index.
        pivot_index: The index of the pivot element.
        key: A function to extract a comparison key from elements.
    
    Returns:
        int: The final position of the pivot element.
    """
    high -= 1  # make it inclusive

    # exception case: if one competition is enough
    if high - low == 1: # if the sequence has only 2 elements
        if key(seq[low]) > key(seq[high]):
            seq[low], seq[high] = seq[high], seq[low]
        return high

    seq[pivot_index], seq[high] = seq[high], (pivot_value := seq[pivot_index])
    left, right = low, high - 1  # setting left and right indices (pivot excluded)

    # to prevent repeating item access and function calls
    left_value, right_value, pivot_key = seq[left], seq[right], key(pivot_value)

    while left < right:
        while key(left_value) < pivot_key:
            left_value = seq[left := left + 1]

        while pivot_key < key(right_value):
            right_value = seq[right := right - 1]

        if left < right:
            seq[left], seq[right] = right_value, left_value
            left_value, right_value = seq[left := left + 1], seq[right := right - 1]

    seq[left], seq[high] = pivot_value, left_value
    
    return left


# optimized versions (considering all inputs are valid, no pythonic indexing)
# a few defaults are set to make it easier to use manually

def _iterative_quick_sort(
    seq: abc.MutableSequence[T],
    /,
    low: int,
    high: int,
    *,
    k: int = 0,  # max distance from the sorted position
    pivot_selector: abc.Callable[[int], int] = middle_as_pivot,
    key: Key[T] = identity,
) -> None:
    """Performs iterative quicksort on a sequence.
    
    Args:
        seq: The sequence to sort.
        low: The lower bound index.
        high: The upper bound index.
        k: Maximum distance from sorted position.
        pivot_selector: Function to select pivot index.
        key: Function to extract comparison key from elements.
    """
    length = high - low

    if length < 2:  # base case: only one element or empty
        return # already counts sorted

    k += 1  # k = minimum required elements to sort - 1

    # stack is used to avoid recursion and stack overflow
    size = length >> 1  # covers minimum stack size
    size += size & 1    # makes it even to use odd/even trick with no waste of space
    stack = [0] * size  # auxiliary stack

    pointer = 0  # stack pointer
    # pointer is an even number pointing to low.
    # pointer + 1 points to high.
    # helps to avoid multiplication by 2 or two times addition ot 1.

    stack[0] = low  # low is even
    stack[1] = high  # high is odd

    while pointer >= 0:  # while stack is not empty
        # gets high and low from stack (no pop needed because of the pointer)
        low = stack[pointer]
        high = stack[pointer + 1]
        
        # considering the pivot_selector(...) is always valid
        pivot_index = partition(seq, low, high, pivot_selector(high - low) + low , key)

        left = pivot_index - low > k  # does the left side have mode than k elements to sort?
        right = high - pivot_index > k  # does the right side have at least two elements to sort?
        
        if left and right:
            stack[pointer] = low  # low is even
            stack[pointer + 1] = pivot_index  # pivot_index is odd
            pointer += 2  # moves to the next pair of low and high
            
            if pointer < size - 1:  # if there is space in the stack
                stack[pointer] = pivot_index  # pivot_index + 1 is even
                stack[pointer + 1] = high  # high is odd
            else:
                stack.extend((pivot_index, high))  # appends the new pair
                size += 2  # increases the stack size
        
        elif left or right:
            stack[pointer] = low if left else pivot_index  # low is even or pivot_index + 1 is even
            stack[pointer + 1] = pivot_index if left else high  # high is odd or pivot_index is odd
        
        else:
            pointer -= 2

def _quick_sort(
    seq: abc.MutableSequence[T],
    /,
    low: int,
    high: int,
    *,
    k: int = 0,  # max distance from the sorted position
    pivot_selector: abc.Callable[[int], int] = middle_as_pivot,
    key: Key[T] = identity,
) -> None:
    """Performs recursive quicksort on a sequence.
    
    Args:
        seq: The sequence to sort.
        low: The lower bound index.
        high: The upper bound index.
        k: Maximum distance from sorted position.
        pivot_selector: Function to select pivot index.
        key: Function to extract comparison key from elements.
    """
    length = high - low
    if length > k + 1:  # when there are enough elements to sort
        # considering the pivot(...) is always valid
        pivot_index = partition(seq, low, high, pivot_selector(length) + low, key)
        
        _quick_sort(seq, low, pivot_index, pivot_selector=pivot_selector, key=key, k=k)
        _quick_sort(seq, pivot_index + 1, high, pivot_selector=pivot_selector, key=key, k=k)


def _quick_select(
    seq: abc.MutableSequence[T],
    /,
    low: int,
    high: int,
    k: int = 0,
    *,
    pivot_selector: abc.Callable[[int], int] = middle_as_pivot,
    key: Key[T] = identity,
) -> T:
    """Selects the kth smallest element in a sequence using quickselect algorithm.
    
    Args:
        seq: The sequence to search.
        low: The lower bound index.
        high: The upper bound index.
        k: The rank of the element to find (0-based).
        pivot_selector: Function to select pivot index.
        key: Function to extract comparison key from elements.
    
    Returns:
        The kth smallest element in the sequence.
    """
    length = high - low

    if length == 1:
        return seq[low]

    while True:
        pivot_index = pivot_selector(length) + low  # pivot is always in the range of low and high
        pivot_index = partition(seq, low, high, pivot_index, key)

        # pivot_index is the index of the pivot in the sorted array
        if pivot_index == k:
            return seq[pivot_index]

        if pivot_index < k:
            low = pivot_index + 1
        
        else:
            high = pivot_index - 1  # doesn't include the pivot in the next iteration
        
        if (length := high - low) == 1:  # updates the length and checks if it's 1
            return seq[low]


# safe versions (with starting validations and pythonic indexing)
@tp.overload
def quick_sort(
    seq: abc.MutableSequence[T],
    /,
    low: int = 0,
    high: int = ...,
    *,
    k: int = 0,
    pivot_selector: abc.Callable[[int], int] = middle_as_pivot,
    key: Key[T] = identity,
    iterative: bool = False,
) -> None: ...

@tp.overload
def quick_sort(
    seq: abc.MutableSequence[T],
    /,
    low: int = 0,
    high: int = ...,
    *,
    k: int = 0,
    pivot_selector: abc.Callable[[int, int], list[int]] = None,
    key: Key[T] = identity,
    iterative: bool = False,
) -> None: ...


def quick_sort(
    seq, /, low=0, high=None, *, k=0,
    pivot_selector=None, key=identity, iterative=False,
) -> None:
    """Sort a mutable sequence in-place using quicksort algorithm.

    Args:
        seq: Mutable sequence to sort
        low: Starting index for sorting (default 0)
        high: Ending index for sorting (default the end of the sequence)
        k: Minimum size of partition for recursion (default 0)
        pivot_count: Number of pivots to use (default 1)
        pivot_selector: Function to select pivot(s) (default None)
        key: Function to extract comparison key (default identity)
        iterative: Whether to use iterative implementation (default False)
    """
    for check, typ, name in (
        (isinstance(seq, abc.MutableSequence), TypeError, 'seq must be a mutable sequence'),
        (isinstance(low, int), TypeError, 'low must be an integer'),
        (high is None or isinstance(high, int), TypeError, 'high must be an integer'),
        (isinstance(k, int), TypeError, 'k must be an integer'),
        (k >= 0, ValueError, 'k must be non-negative'),
        (pivot_selector is None or callable(pivot_selector),
         TypeError, 'pivot_selector must be callable'),
        (callable(key), TypeError, 'key must be callable'),
        (isinstance(iterative, bool), TypeError, 'iterative must be a boolean'),
    ):
        if not check:
            raise typ(f'{name}, got {locals()[name.split()[0]]} instead')

    if (length := len(seq)) < 2:
        return # already counts sorted
    
    low = positive_index(low, length)
    high = positive_index(high, length)
    
    if high - low < 2 + k:
        return # already counts sorted

    (_iterative_quick_sort if iterative else _quick_sort)(
        seq, low, high, k=k, key=key,
        pivot_selector=middle_as_pivot if pivot_selector is None else pivot_selector,  # type: ignore
    )


def quick_select(
    seq: abc.MutableSequence[T],
    /,
    k: int,
    low: int = 0,
    high: int = -1,
    *,
    pivot_selector: abc.Callable[[int], int] = middle_as_pivot,
    key: Key[T] = identity,
) -> T:
    """Select the kth smallest element from a mutable sequence using quickselect algorithm.

    Args:
        seq: Mutable sequence to select from
        k: Index of element to select (0-based)
        low: Starting index for selection (default 0)
        high: Ending index for selection (default -1)
        pivot_selector: Function to select pivot (default middle_as_pivot)
        key: Function to extract comparison key (default identity)

    Returns:
        The kth smallest element in the sequence

    Raises:
        ValueError: If sequence is empty
        IndexError: If k is out of range
    """
    for check, typ, name in (
        (isinstance(seq, abc.MutableSequence), TypeError, 'seq must be a mutable sequence'),
        (isinstance(low, int), TypeError, 'low must be an integer'),
        (isinstance(high, int), TypeError, 'high must be an integer'),
        (isinstance(k, int), TypeError, 'k must be an integer'),
        (k >= 0, ValueError, 'k must be non-negative'),
        (callable(pivot_selector), TypeError, 'pivot_selector must be callable'),
        (callable(key), TypeError, 'key must be callable'),
    ):
        if not check:
            raise typ(f'{name}, got {locals()[name.split()[0]]} instead')

    if (length := len(seq)) < 1:
        raise ValueError('empty sequence')
    
    low = positive_index(low, length)
    high = positive_index(high, length)

    if k < low or high <= k:
        raise IndexError('index out of range')

    return _quick_select(seq, low, high, k, pivot_selector=pivot_selector, key=key)
